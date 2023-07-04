import modules.scripts as scripts
import gradio as gr
import os
import tqdm
import torch
from torch.optim import Adam, AdamW, SGD, Adadelta, Adagrad, SparseAdam, Adamax, ASGD, LBFGS, NAdam, RAdam, RMSprop, Rprop
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,OneCycleLR,CyclicLR,ReduceLROnPlateau,SequentialLR,ChainedScheduler,CosineAnnealingLR,PolynomialLR,ExponentialLR,LinearLR,ConstantLR,MultiStepLR,MultiStepLR,MultiplicativeLR,LambdaLR
from torch.nn import L1Loss,MSELoss,CrossEntropyLoss,CTCLoss,NLLLoss,PoissonNLLLoss,GaussianNLLLoss,KLDivLoss,BCELoss,BCEWithLogitsLoss,MarginRankingLoss,HingeEmbeddingLoss,MultiLabelMarginLoss,HuberLoss,SmoothL1Loss,SoftMarginLoss,MultiLabelSoftMarginLoss,CosineEmbeddingLoss,MultiMarginLoss,TripletMarginLoss,TripletMarginWithDistanceLoss

import modules
from modules import devices, shared, script_callbacks, prompt_parser, sd_hijack
from modules.textual_inversion.textual_inversion import Embedding
from modules.shared import opts, cmd_opts

from lion_pytorch import Lion
from prodigyopt import Prodigy

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui_component:
        with gr.Row():
            gr_step = gr.Slider(
                minimum=0,
                maximum=150,
                step=1,
                value=20,
                label="Steps"
            ) 

            gr_layer = gr.Slider(
                minimum=1,
                maximum=75,
                step=1,
                value=1,
                label="layer"
            )

            # TODO: add more UI components (cf. https://gradio.app/docs/#components)
        with gr.Row():
            gr_text = gr.Textbox(
                value='', 
                lines=4, 
                max_lines=16, 
                interactive=True, 
                label='Your prompt'
            )
        
        with gr.Row():
            gr_late = gr.Number(
                value = 0.0001,
                maximum = 1.0,
                minimum = 0.0,
                label="learning_late"
            )
        with gr.Row():
            gr_optimizer = gr.Dropdown(
                choices=('Adam','AdamW','Lion','Prodigy','SGD','Adadelta','Adagrad','SparseAdam','Adamax','ASGD','LBFGS','NAdam','RAdam','RMSprop','Rprop'), 
                value='Adam', 
                type='value', 
                interactive=True, 
                label='Optimizer'
            )

            gr_loss = gr.Dropdown(
                choices=('L1Loss','MSELoss','CrossEntropyLoss','CTCLoss','NLLLoss','PoissonNLLLoss','GaussianNLLLoss','KLDivLoss','BCELoss','BCEWithLogitsLoss','MarginRankingLoss','HingeEmbeddingLoss','MultiLabelMarginLoss','HuberLoss','SmoothL1Loss','SoftMarginLoss','MultiLabelSoftMarginLoss','CosineEmbeddingLoss','MultiMarginLoss','TripletMarginLoss','TripletMarginWithDistanceLoss'), 
                value='MSELoss', 
                type='value', 
                interactive=True, 
                label='loss'
            )

            gr_scheduler = gr.Dropdown(
                choices=('CosineAnnealingWarmRestarts','OneCycleLR','CyclicLR','ReduceLROnPlateau','SequentialLR','ChainedScheduler','CosineAnnealingLR','PolynomialLR','ExponentialLR','LinearLR','ConstantLR','MultiStepLR','MultiStepLR','MultiplicativeLR','LambdaLR'), 
                value='LambdaLR', 
                type='value', 
                interactive=True, 
                label='scheduler'
            )
        
        with gr.Row():
            gr_lstep = gr.Number(
                value = 100,
                maximum = 100000,
                minimum = 0,
                label="learning_step"
            )

        with gr.Row():
            gr_init = gr.Textbox(
                value='', 
                lines=1, 
                max_lines=1, 
                interactive=True, 
                label='initial prompt. (If blank, the initial value is random. If not blank, the number of LAYERS is automatically changed to the number of tokens in the input prompt.)'
            )
            gr_layerow = gr.Checkbox(
                label='The number of tokens can be overridden by the number of LAYERS.'
            ) 

        with gr.Row():
            gr_name = gr.Textbox(
                value='', 
                lines=1, 
                max_lines=1, 
                interactive=True, 
                label='save embedding name'
            )
            gr_nameow = gr.Checkbox(
                label='enable over write embedding'
            ) 
        
        with gr.Row():
            gr_button = gr.Button(
                'Estimate!',
                variant='primary'
            )
            gr_interrupt = gr.Button("Interrupt", elem_id="train_interrupt_preprocessing")
        
        #interrupt training
        gr_interrupt.click(
            fn=gr_interrupt_train,
            inputs=[],
            outputs=[],
        )
        
        # gr_button.click(fn=gr_func, inputs=[gr_name,gr_text,gr_optimizer,gr_true], outputs=[gr_html,gr_name,gr_text], show_progress=False)
        gr_button.click(fn=gr_func, inputs=[gr_text,gr_optimizer,gr_loss,gr_scheduler,gr_step,gr_layer,gr_late,gr_lstep,gr_init,gr_layerow,gr_name,gr_nameow], show_progress=False)

    return [(ui_component, "Embedding Estimate", "extension_template_tab")]

def gr_func(gr_text,gr_optimizer,gr_loss,gr_scheduler,gr_step,gr_layer,gr_late,gr_lstep,gr_init,gr_layerow,gr_name,gr_nameow):

    if gr_name == '':
        print("Please write save name")
        return ''
    
    # 入力データの初期化（Nx768次元の配列）
    if gr_init != '' and gr_init is not None:
        clip = shared.sd_model.cond_stage_model

        part = clip.tokenize_line(gr_init)

        cnt = part[1]
    
        # trans = clip.encode_embedding_init_text(gr_init,cnt)

        before_emb = None
        all_vector = torch.zeros(cnt, 768).to(device=devices.device,dtype=torch.float32)

        #　embeddingとtokenを分離と変換後、同位置で再結合
        for count,emb in enumerate(part[0][0].fixes):
            
            vec = emb.embedding.vec
            start = emb.offset
            end = emb.embedding.vectors + start
            name = emb.embedding.name

            if count == 0: #初回処理
                if start == 0:
                    all_vector[start:end] = vec
                else: #左に単語があればそれを代入
                    all_vector[:start] = clip.encode_embedding_init_text(gr_init.split(name)[0],start-1)
                    all_vector[start:end] = vec
            else:
                before_start = before_emb.offset
                before_end = before_emb.embedding.vectors + before_start
                before_name = before_emb.embedding.name
                if start - before_end == 0: #間に単語がなければ今のembeddingだけを代入
                    all_vector[start:end] = vec
                else:
                    between_words = gr_init.split(before_name, 1)[1].split(name, 1)[0]
                    all_vector[before_end:start] = clip.encode_embedding_init_text(between_words,start-before_end)
                    
                    if count == len(part[0][0].fixes): #最終処理
                        all_vector[start:end] = vec
                        if end != cnt: #右に単語があればそれを代入
                            all_vector[end:cnt] = clip.encode_embedding_init_text(gr_init.split(name)[1],cnt-end)

            before_emb = emb 
        
        if gr_layerow:
            input_tensor = all_vector[:gr_layer,:].unsqueeze(0).to(device=devices.device,dtype=torch.float32).requires_grad_(True)
        else:
            input_tensor = all_vector.unsqueeze(0).to(device=devices.device,dtype=torch.float32).requires_grad_(True)

    else:
        input_tensor = torch.randn(1, gr_layer, 768, requires_grad=True)  # 入力テンソルに勾配を追跡させる


    # 関数に渡す引数（辞書）
    optimizer_arg = {
        'params':[input_tensor],
        'lr': gr_late
    }  
    
    # オプティマイザの選択
    if gr_optimizer in globals():
        optimizer_function = globals()[gr_optimizer]
        optimizer = optimizer_function(**optimizer_arg)

        
    else:
        print("The specified optimizer does not exist.")
    
    # 損失関数の定義
    if gr_loss in globals():
        loss_function = globals()[gr_loss]
        loss_fn = loss_function()
    else:
        print("The specified loss_function does not exist.")
    

    # 関数に渡す引数（辞書）
    scheduler_arg = {
        'optimizer':[optimizer]
    }  

    # スケジューラの選択
    if gr_scheduler in globals():
        scheduler_function = globals()[gr_scheduler]
        scheduler = scheduler_function(**scheduler_arg)
        if gr_optimizer == "Prodigy":
            # n_epoch is the total number of epochs to train the network
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=gr_lstep/100)


    # 目的出力
    target_output = prompt_parser.get_learned_conditioning(shared.sd_model, [gr_text], gr_step)

    EMBEDDING_NAME = 'embedding_estimate'
    cache = {}
    learning_step = int(gr_lstep)
    
    # 勾配降下法による最適化ループ
    for i in tqdm.tqdm(range(learning_step)):
        def closure():
            optimizer.zero_grad()  # 勾配を初期化
            
            # 入力データからembedingを作成
            make_temp_embedding(EMBEDDING_NAME,input_tensor.squeeze(0).to(device=devices.device,dtype=torch.float16),cache) #ある番号ごとに保存機能も後で追加か

            output = prompt_parser.get_learned_conditioning(shared.sd_model, [EMBEDDING_NAME], gr_step) # 入力テンソルをモデルに通す -> embeding登録してプロンプトから通す
            # output = model.get_text_features(input_tensor)  
            
            loss = loss_fn(output[0][0].cond, target_output[0][0].cond)  # 損失を計算
            loss.backward()  # 勾配を計算

            if i % 100 == 0 or i == learning_step:
                print(f"\nIteration {i}, Loss: {loss.item()}")
            
            return loss
        
        optimizer.step(closure)
        scheduler.step()

        global stop_training
        
        if stop_training:
            print("Interrupt training")
            stop_training = False
            break

    print("Training completed!")
    
    need_save_embed(gr_name,input_tensor.squeeze(0),gr_nameow)

    print("embedding save is finished!")


stop_training = False

def gr_interrupt_train():
    global stop_training 
    stop_training = True

def make_temp_embedding(name,vectors,cache):
    if name in cache:
        embed = cache[name]
    else:
        embed = Embedding(vectors,name)
        cache[name] = embed
    embed.vec = vectors
    embed.step = None
    shape = vectors.size()
    embed.vectors = shape[0]
    embed.shape = shape[-1]
    embed.cached_checksum = None
    embed.filename = ''
    register_embedding(name,embed)

def register_embedding(name,embedding):
    # /modules/textual_inversion/textual_inversion.py
    self = modules.sd_hijack.model_hijack.embedding_db
    model = shared.sd_model
    try:
        ids = model.cond_stage_model.tokenize([name])[0]
        first_id = ids[0]
    except:
        return
    if embedding is None:
        if self.word_embeddings[name] is None:
            return
        del self.word_embeddings[name]
    else:
        self.word_embeddings[name] = embedding
    if first_id not in self.ids_lookup:
        if embedding is None:
            return
        self.ids_lookup[first_id] = []
    save = [(ids, embedding)] if embedding is not None else []
    old = [x for x in self.ids_lookup[first_id] if x[1].name!=name]
    self.ids_lookup[first_id] = sorted(old + save, key=lambda x: len(x[0]), reverse=True)
    return embedding

merge_dir = None
def need_save_embed(name,vectors,ow):
    name = ''.join( x for x in name if (x.isalnum() or x in '._- ')).strip()
    if name=='':
        return name
    try:
        if type(vectors)==list:
            vectors = torch.cat([r[0] for r in vectors])
        file = modules.textual_inversion.textual_inversion.create_embedding('_EmbeddingMerge_temp',vectors.size(0),ow,init_text='')
        pt = torch.load(file,map_location='cpu')
        token = list(pt['string_to_param'].keys())[0]
        pt['string_to_param'][token] = vectors.cpu()
        torch.save(pt,file)
        merge_dir = embedding_merge_dir()
        target = os.path.join(merge_dir,name+'.pt')
        os.replace(file,target)
        modules.sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()
        return ''
    except:
        # traceback.print_exc()
        return name

def embedding_merge_dir():
    try:
        merge_dir = os.path.join(cmd_opts.embeddings_dir,'embedding_estimate')
        modules.sd_hijack.model_hijack.embedding_db.add_embedding_dir(merge_dir)
        os.makedirs(merge_dir)
    except:
        pass

    return merge_dir

script_callbacks.on_ui_tabs(on_ui_tabs)


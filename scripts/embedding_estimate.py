import modules.scripts as scripts
import gradio as gr
import os
import torch
from torch.optim import Adam, LBFGS
from torch.nn import MSELoss

import modules
from modules import devices, shared, script_callbacks, prompt_parser, sd_hijack
from modules.textual_inversion.textual_inversion import Embedding
from modules.shared import opts, cmd_opts

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui_component:
        with gr.Row():
            gr_step = gr.Slider(
                minimum=0,
                maximum=150,
                step=1,
                value=0,
                label="Steps"
            )
            checkbox = gr.Checkbox(
                False,
                label="Checkbox"
            )
        with gr.Row():
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
                label='Your prompt (no weight/attention, do not escape parenthesis/brackets); or your merge expression (if the first character is a single quote); or a generation info to restore prompts'
            )
        
        with gr.Row():
            gr_late = gr.Number(
                value = 0.0001,
                maximum = 1.0,
                minimum = 0.0,
                label="learning_late"
            )
        with gr.Row():
            gr_radio = gr.Radio(
                choices=('Adam'), 
                value='By parts', 
                type='index', 
                interactive=True, 
                label='Group/split table by: (when not started with single quote - so only for prompts, not for merge)'
            )
        
        with gr.Row():
            gr_lstep = gr.Number(
                value = 100,
                maximum = 100000,
                minimum = 0,
                label="learning_step"
            )
        
        with gr.Row():
            gr_button = gr.Button(
                'Estimate!',
                variant='primary'
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
        
        # gr_button.click(fn=gr_func, inputs=[gr_name,gr_text,gr_radio,gr_true], outputs=[gr_html,gr_name,gr_text], show_progress=False)
        gr_button.click(fn=gr_func, inputs=[gr_text,gr_radio,gr_step,gr_layer,gr_late,gr_lstep,gr_name,gr_nameow], show_progress=False)

    return [(ui_component, "Extension Template", "extension_template_tab")]

def gr_func(gr_text,gr_radio,gr_step,gr_layer,gr_late,gr_lstep,gr_name,gr_nameow):

    if gr_name is None:
        print("Please write save name")
        return ''

    # 入力データの初期化（Nx768次元の配列）
    input_tensor = torch.randn(1, gr_layer, 768, requires_grad=True)  # 入力テンソルに勾配を追跡させる

    # .squeeze(0).to(device=devices.device,dtype=torch.float16)

    # input_tensor = torch.randn(1, gr_layer, 768).squeeze(0).to(device=devices.device,dtype=torch.float16)  # 入力テンソルに勾配を追跡させる
    input_tensor_opt = input_tensor.detach().requires_grad_(True)

    # オプティマイザの選択
    if gr_radio == 'Adam':
        optimizer = Adam([input_tensor], lr=gr_late)
    else:
        optimizer = Adam([input_tensor], lr=gr_late)
    
    # 損失関数の定義
    loss_fn = MSELoss()
    
    # 目的出力
    target_output = prompt_parser.get_learned_conditioning(shared.sd_model, [gr_text], gr_step)

    # estimate_vector_value(input_tensor, target_output, loss_fn, optimizer, gr_lstep, gr_step,gr_name)

    # print(f"{input_tensor[0][0].cond}")

# def estimate_vector_value(input_tensor, target_output, loss_fn, optimizer, gr_lstep, gr_step, gr_name):
    
    EMBEDDING_NAME = 'embedding_estimate'
    cache = {}
    
    # 勾配降下法による最適化ループ
    for i in range(int(gr_lstep)):
        def closure():
            optimizer.zero_grad()  # 勾配を初期化
            
            # 入力データからembedingを作成
            make_temp_embedding(EMBEDDING_NAME,input_tensor.squeeze(0).to(device=devices.device,dtype=torch.float16),cache) #ある番号ごとに保存機能も後で追加か

            output = prompt_parser.get_learned_conditioning(shared.sd_model, [EMBEDDING_NAME], gr_step) # 入力テンソルをモデルに通す -> embeding登録してプロンプトから通す
            # output = model.get_text_features(input_tensor)  
            
            loss = loss_fn(output[0][0].cond, target_output[0][0].cond)  # 損失を計算
            loss.backward()  # 勾配を計算

            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss.item()}")
            
            return loss
        
        optimizer.step(closure)

    print("Training completed!")
    
    need_save_embed(gr_name,input_tensor.squeeze(0),gr_nameow)

    print("embedding save is finished!")


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

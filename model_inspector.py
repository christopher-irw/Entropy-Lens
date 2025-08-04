import transformer_lens
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils import get_entropy, get_renyi_entropy, get_moment, kl_divergence, get_entropy_thresholded, get_moment_thresholded
import einops as ein

class ModelInspector():
    def __init__(self, model_name, device='cuda', dtype='float32'):
        self.model = transformer_lens.HookedTransformer.from_pretrained(model_name, 
                                                                        center_unembed=False, 
                                                                        fold_ln=False, 
                                                                        center_writing_weights=False,
                                                                        dtype=dtype
                                                                        ).to(device)
        self.model_name = model_name
        self.device = device
        self.set_model_to_eval()
        self.weight_tying = self.check_weight_tying()
        self.scaled_layers = {}
        self.soft_cap = self.model.cfg.output_logits_soft_cap
        

    
    def set_model_to_eval(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        print('Model set to eval mode')
        return
    
    def check_weight_tying(self):
        tyed = torch.allclose(self.model.embed.W_E, self.model.unembed.W_U.T)
        print(f'Weight tying: {tyed}')
        return tyed
    
    @torch.no_grad()
    def get_all_hidden_states(self, activations, layer_name='hook_resid_post'):
        return torch.vstack([activations[f'blocks.{l}.{layer_name}'].detach()[:,-1] for l in range(len(self.model.blocks))]).unsqueeze(0)


    @torch.no_grad()
    def generate_with_activations(self, prompt, max_len=100, layer_name='hook_resid_post', verbose=True, sample=False, temperature=1.0):
        """
        Generates text with activations from a given prompt.
        Args:
            prompt (str): The initial text prompt to start generation.
            max_len (int, optional): The maximum length of the generated text. Defaults to 100.
            layer_name (str, optional): The name of the layer from which to extract activations. Defaults to 'hook_resid_post'.
        Returns:
            tuple: A tuple containing the generated text (str) and a tensor of all activations (torch.Tensor).
        """

        all_activations = []
        pbar = tqdm(range(max_len)) if verbose else range(max_len)

        for i in pbar:
            with torch.no_grad():
                logits, activations = self.model.run_with_cache(prompt)
            all_activations.append(self.get_all_hidden_states(activations, layer_name))
            
            if sample:
                probs = F.softmax(logits/temperature, -1)
                new_tok = torch.multinomial(probs[:,-1], 1)
                new_tok = self.model.to_string(new_tok)[0]
            else:
                new_tok = self.model.to_string((logits).softmax(-1).argmax(-1)[:,-1])

            if self.model_name == 'phi-3':
                prompt += ' ' + new_tok
            else:
                prompt += new_tok

        return prompt, torch.vstack(all_activations)
    
    def unembed(self, activations):
        logits = self.model.unembed(self.model.ln_final(activations))
        if self.soft_cap > .0:
            logits = self.soft_cap * torch.tanh(logits / self.soft_cap)
        return logits

    @torch.no_grad()
    def decode_activations(self, activations):
        """
        Decodes the given activations into human-readable text.
        Args:
            activations (torch.Tensor): The activations to decode.
        Returns:
            List[str]: A list of decoded text strings.
        """

        pred_toks = self.unembed(activations).cpu().softmax(-1).argmax(-1)
        pred_txt = [self.model.to_string(pred_toks[i].view(-1, 1)) for i in range(pred_toks.shape[0])]
        
        return pred_txt
    
    @torch.no_grad()
    def calculate_entropy(self, activations, num_windows=10):
        """
        Calculate the entropy of the given activations over a specified number of windows.
        Args:
            activations (torch.Tensor): The activations from the model, expected to be a 2D tensor.
            num_windows (int, optional): The number of windows to divide the activations into. Default is 10.
        Returns:
            torch.Tensor: The mean entropy calculated over the specified number of windows.
        Raises:
            AssertionError: If the number of windows does not divide the number of tokens in activations.
        """

        assert activations.shape[0] % num_windows == 0, 'Number of windows must divide the number of tokens'
        entropy = get_entropy(self.unembed(activations).cpu().float())
        entropy = ein.rearrange(entropy, '(w t) l -> w t l', l=len(self.model.blocks), w=num_windows).mean(1)
        return entropy

    @torch.no_grad()
    def calculate_renyi_entropy(self, activations, alpha, num_windows=10):
        """
        Calculate the renyi entropy of the given activations over a specified number of windows.
        Args:
            activations (torch.Tensor): The activations from the model, expected to be a 2D tensor.
            alpha (float): The alpha parameter for the renyi entropy. Must be >= 0.
            num_windows (int, optional): The number of windows to divide the activations into. Default is 10.
        Returns:
            torch.Tensor: The mean renyi entropy calculated over the specified number of windows.
        Raises:
            AssertionError: If the number of windows does not divide the number of tokens in activations.
            AssertionError: If alpha < 0.
        """
        assert activations.shape[0] % num_windows == 0, 'Number of windows must divide the number of tokens'
        assert alpha >= 0, 'Alpha must be >= 0'
        renyi_entropy = get_renyi_entropy(self.unembed(activations).cpu().float(), alpha=alpha)
        renyi_entropy = ein.rearrange(renyi_entropy, '(w t) l -> w t l', l=len(self.model.blocks), w=num_windows).mean(1)
        return renyi_entropy
    
    @torch.no_grad()
    def calculate_anchored_renyi_entropy(self, activations, anchors=[], alpha=1, num_windows=1):
        """
        Calculate the renyi entropy of the given activations over a specified number of windows and anchors.
        Args:
            activations (torch.Tensor): The activations from the model, expected to be a 2D tensor.
            anchors (List: Str): A list of anchors strings to calculate the entropy against. If empty, it will calculate the entropy over all tokens.
            alpha (float): The alpha parameter for the renyi entropy. Must be >= 0.
            num_windows (int, optional): The number of windows to divide the activations into. Default is 10.
        Returns:
            torch.Tensor: The mean renyi entropy calculated over the specified number of windows.
        Raises:
            AssertionError: If the number of windows does not divide the number of tokens in activations.
            AssertionError: If alpha < 0.
        """
        assert activations.shape[0] % num_windows == 0, 'Number of windows must divide the number of tokens'
        assert alpha >= 0, 'Alpha must be >= 0'

        logits = self.unembed(activations).cpu().float()
        if len(anchors):
            anchors = self.model.to_tokens(anchors).unique().cpu()
            logits = logits[:, :, anchors]

        renyi_entropy = get_renyi_entropy(logits, alpha=alpha)
        renyi_entropy = ein.rearrange(renyi_entropy, '(w t) l -> w t l', l=len(self.model.blocks), w=num_windows).mean(1)
        return renyi_entropy
    
    @torch.no_grad()
    def calculate_moment(self, activations, power, num_windows=10):
        """
        Calculate the entropy of the given activations over a specified number of windows.
        Args:
            activations (torch.Tensor): The activations from the model, expected to be a 2D tensor.
            num_windows (int, optional): The number of windows to divide the activations into. Default is 10.
        Returns:
            torch.Tensor: The mean entropy calculated over the specified number of windows.
        Raises:
            AssertionError: If the number of windows does not divide the number of tokens in activations.
        """

        assert activations.shape[0] % num_windows == 0, 'Number of windows must divide the number of tokens'
        moment = get_moment(self.unembed(activations).cpu().float(), power=power)
        moment = ein.rearrange(moment, '(w t) l -> w t l', l=len(self.model.blocks), w=num_windows).mean(1)
        return moment

    @torch.no_grad()
    def calculate_entropy_batched(self, activations, num_windows=10):
        '''Same as calculate_entropy but with a batched version of the entropy calculation'''
        assert activations.shape[1] % num_windows == 0, 'Number of windows must divide the number of tokens'
        # entropy = get_entropy(self.unembed(activations).cpu().float())
        entropy = get_entropy(self.unembed(activations).float()).cpu()
        entropy = ein.rearrange(entropy, 'b (w t) l -> b w t l', l=len(self.model.blocks), w=num_windows, b=activations.shape[0]).mean(2)
        
        return entropy
    
    @torch.no_grad()
    def calculate_entropy_thresholded(self, activations, num_windows=10, th=None):
        """As calculate_entropy but considers only the logits above a certain threshold"""
        assert activations.shape[0] % num_windows == 0, 'Number of windows must divide the number of tokens'
        entropy = get_entropy_thresholded(self.unembed(activations).cpu().float(), th=th)
        entropy = ein.rearrange(entropy, '(w t) l -> w t l', l=len(self.model.blocks), w=num_windows).mean(1)
        return entropy
    
    @torch.no_grad()
    def calculate_moment_thresholded(self, activations, power, num_windows=10, th=None):
        """As calculate moment but considers only the logits above a certain threshold"""

        assert activations.shape[0] % num_windows == 0, 'Number of windows must divide the number of tokens'
        moment = get_moment_thresholded(self.unembed(activations).cpu().float(), power=power, th=th)
        moment = ein.rearrange(moment, '(w t) l -> w t l', l=len(self.model.blocks), w=num_windows).mean(1)
        return moment

    @torch.no_grad()
    def add_scale_activations_layer(self, block: int, hook: str, scaling_factor: float) -> None:
        """
        Scale the activations of a given transformer block after resid by a specified factor.
        Args:
            block (int): The block index to scale.
            hook (str): The hook name to add the scaling layer to.
            scaling_factor (float): The factor by which to scale the activations.
        """

        hook_name = f'blocks.{block}.{hook}'

        if block in self.scaled_layers:
            raise Exception(f'Block {block} already scaled. Undo scaling before adding new scaling layer')

        scale_hook = lambda activations, hook : scaling_factor * activations
        self.model.add_hook(hook_name, scale_hook, dir="fwd") 
        self.scaled_layers[block] = {'scaling_factor': scaling_factor, 'hook_name': hook_name}

        return
    

    @torch.no_grad()
    def undo_scaling_at_layer(self, block: int) -> None:
        """
        Undo scaling at a given transformer block.
        Args:
            block (int): The block index to undo scaling.
        Raises:
            Exception: If the block is not scaled.
        """
        if block not in self.scaled_layers:
            raise Exception(f'Block {block} not scaled. No scaling to undo')

        data = self.scaled_layers[block]
        prev_scaling_factor = data['scaling_factor']
        hook_name = data['hook_name']
        undo_scaling = lambda activations, hook : activations / prev_scaling_factor
        self.model.add_hook(hook_name, undo_scaling, dir="fwd") 
        del self.scaled_layers[block]
        return
    

    @torch.no_grad()
    def undo_all_scalings(self) -> None:
        """
        Undo all scalings applied to the model.
        """
        for block, data in self.scaled_layers.items():
            prev_scaling_factor = data['scaling_factor']
            undo_scaling = lambda activations, hook : activations / prev_scaling_factor
            self.model.add_hook(data['hook_name'], undo_scaling, dir="fwd") 

        self.scaled_layers = {}
        return
    


if __name__ == '__main__':
    model_name = 'gpt2-xl'          # model to inspect
    layer_name = 'hook_resid_post'  # layer to inspect
    max_len = 16                    # number of tokens to generate
    
    model_inspector = ModelInspector(model_name)
    prompt = 'Which scientist introduced the concept of relativity?'

    # Generate text and keeping track of activations
    prompt, all_activations = model_inspector.generate_with_activations(prompt, max_len, layer_name)
    
    # Decode activations to text splitte by layers
    unrolled_txt = model_inspector.decode_activations(all_activations)
    # in the print statement, we are printing the last tokens of the last 10 layers
    [print(f'token {i}: {txt[-10:]}') for i, txt in enumerate(unrolled_txt)]

    # print the generated text
    print(prompt)
    # print the entropy of the activations, with a rolling average over 'num_windows'
    print(model_inspector.calculate_entropy(all_activations, num_windows=1))


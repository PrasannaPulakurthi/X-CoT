import torch
import torch.nn as nn
from config.base_config import Config
from modules.transformer import Transformer, AvgPooling
from model.vlm2vec_src.model import MMEBModel
from model.vlm2vec_src.arguments import ModelArguments
from model.vlm2vec_src.model_utils import load_processor, QWEN2_VL, vlm_image_tokens
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

class Qwen2VLAdvancePromptRetrieval(nn.Module):
    def __init__(self, config: Config):
        super(Qwen2VLAdvancePromptRetrieval, self).__init__()
        self.config = config

        # Use Qwen2VL 
        model_args = ModelArguments(
            model_name='Qwen/Qwen2-VL-7B-Instruct',
            checkpoint_path='TIGER-Lab/VLM2Vec-Qwen2VL-7B',
            pooling='last',
            normalize=True,
            model_backbone='qwen2_vl',
            lora=True
        )
        self.processor = load_processor(model_args)
        self.llm = MMEBModel.load(model_args)

        if self.config.gpu != '99':
            device = torch.device(f'cuda:{config.gpu}')
            self.llm = self.llm.to(device).to(torch.bfloat16)
        else:
            # Let PyTorch automatically choose an available GPU
            self.llm = self.llm.cuda().to(torch.bfloat16)
        self.llm.eval()  # Use as a feature extractor

        if config.pooling_type == 'transformer':
            self.pool_frames = Transformer(config).to(torch.bfloat16)
        elif config.pooling_type == 'avg':
            self.pool_frames = AvgPooling(config).to(torch.bfloat16)

    def get_frame_prompt(self, frame_idx, num_frames):
        """
        Returns a comprehensive prompt for video frame analysis that focuses on:
        1. Main subjects and their actions
        2. Scene context and dynamics
        3. Key visual elements for text matching
        """
        if self.config.dataset_name == 'MSRVTT':
            prompt = (
                f"Imagine this freeze-frame {vlm_image_tokens[QWEN2_VL]} from a video must be found among thousands. " 
                "Identify the main subject(s) and their actions or expressions, noting any crucial objects or distinct background details. " 
                "Keep it concise and highlight whatever makes this exact moment stand out, ensuring accurate text-to-video matching." 
            )
        elif self.config.dataset_name == 'DiDeMo':
            prompt = (
                f"Describe what is happening in this frame {vlm_image_tokens[QWEN2_VL]} by focusing on the main subject's action. " 
                "Mention any movements or interactions that define this moment in the video. " 
                "Keep it concise and highlight whatever makes this exact moment stand out, ensuring accurate text-to-video matching." 
            )
        elif self.config.dataset_name == 'LSMDC':
            prompt = (
                f"Analyze this frame {vlm_image_tokens[QWEN2_VL]}and describe the action occurring in this moment. "
                "Identify the physical movement of characters, if they are running, fighting, gesturing, or engaging in any motion. "
                "Focus on dynamic elements such as body posture, limb placement, or any interactions with objects that define the action."
            )
        elif self.config.dataset_name == 'MSVD':
            prompt = (
                f"Analyze this frame {vlm_image_tokens[QWEN2_VL]} within the context of a video. "
                "Describe the main subject's motion, their interaction with objects, and how the surrounding environment influences the action. "
                "Focus on the momentary activity while considering how this frame fits into the larger sequence."
            )
        elif self.config.dataset_name == 'CHARADES':
            prompt = (
                f"Imagine this freeze-frame {vlm_image_tokens[QWEN2_VL]} from a video must be found among thousands. " 
                "Identify the main subject(s) and their actions or expressions, noting any crucial objects or distinct background details. " 
                "Keep it concise and highlight whatever makes this exact moment stand out, ensuring accurate text-to-video matching." 
            )
        else:
            raise NotImplementedError
        
        prompt = (f"Represent the given image {vlm_image_tokens[QWEN2_VL]}.")
        # prompt = (f"{vlm_image_tokens[QWEN2_VL]} Represent the given image with the following question: What is in the image")
        return prompt

    def get_text_prompt(self, caption):
        """
        Returns a structured prompt for text caption processing.
        Args:
            caption (str): Original video caption
        """
        if self.config.dataset_name == 'MSRVTT':
            prompt = f"Find the video that best matches the description: '{caption}'. Identify what stands out."
        elif self.config.dataset_name == 'DiDeMo':
            prompt = f"Find the best matching video for this scene: '{caption}'. Focus on essential visual elements."
        elif self.config.dataset_name == 'LSMDC':
            prompt = (
                f"Find the scene that best matches this description: '{caption}'. "
                "Identify key movements, gestures, and how the subject interacts with their surroundings. "
                "Consider how body language, facial expressions, and spatial relationships contribute to the scene’s meaning."
            )
        elif self.config.dataset_name == 'MSVD':
            prompt = (
                f"Find the best matching video moment for this description: '{caption}'. "
                "Identify key subjects, their specific actions, and any interactions with objects or the environment. "
                "Consider how the movement evolves over time and how this moment fits within the larger sequence."
            )
        elif self.config.dataset_name == 'CHARADES':
            prompt = f"Find the video that best matches the description: '{caption}'. Identify what stands out."
        else:
            raise NotImplementedError

        prompt = (f"Find the best matching video for this scene: '{caption}'. Focus on essential visual elements.")
        return prompt

    def encode_text(self, text_batch):
        """
        Encodes a batch of text inputs into embeddings using the LLM.
        
        Args:
            text_batch (list): List of text captions.

        Returns:
            torch.Tensor: Encoded text features with shape (batch_size, embed_dim).
        """
        device = next(self.llm.parameters()).device  

        text_inputs = self.processor(
            text=[self.get_text_prompt(cap) for cap in text_batch], 
            images=None,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        text_features = self.llm(tgt=text_inputs)["tgt_reps"]
        # print(f'>>>text_features={text_features.shape}')  # Debugging output
        return text_features

    def encode_video(self, video_batch, return_all_frames=False):
        """
        Encodes a batch of videos into feature embeddings.

        Args:
            video_batch (torch.Tensor): Tensor of shape (batch_size, num_frames, C, H, W).
            return_all_frames (bool): If True, returns both full video features and pooled features.

        Returns:
            torch.Tensor: Encoded video features (batch_size, embed_dim).
        """
        device = next(self.llm.parameters()).device  
        dtype = next(self.llm.parameters()).dtype  

        batch_size, num_frames, C, H, W = video_batch.shape
        video_batch = video_batch.to(device)

        frame_features = []
        for i in range(batch_size):
            for j in range(num_frames):
                # frame_scaled = video_batch[i, j] * 255
                # frame_pil = transforms.ToPILImage()(frame_scaled.cpu().to(torch.uint8))
                frame_pil = to_pil_image(video_batch[i, j])
                
                frame_input = self.processor(
                    text=self.get_frame_prompt(j, num_frames),
                    images=frame_pil,  
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                frame_input = {key: value.to('cuda') for key, value in frame_input.items()}
                # frame_input = {k: v.to(device) for k, v in frame_input.items()}
                frame_input['pixel_values'] = frame_input['pixel_values'].unsqueeze(0)
                frame_input['image_grid_thw'] = frame_input['image_grid_thw'].unsqueeze(0)
                frame_feat = self.llm(qry=frame_input)["qry_reps"]
                frame_features.append(frame_feat)

        video_features = torch.cat(frame_features).reshape(batch_size, num_frames, -1).to(dtype) # [B, F, 3584]
        # print(f'>>>video_features={video_features.shape}')  # Debugging output
        
        text_features = []  # Placeholder if needed by pool_frames
        video_features_pooled, _ = self.pool_frames(text_features, video_features)

        if return_all_frames:
            return video_features, video_features_pooled

        return video_features_pooled

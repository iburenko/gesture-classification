#from timesformer.models.vit import TimeSformer
import pytorch_lightning as pl
import torch
import numpy as np
from torch import nn
from torchvision import models
from transformers import VideoMAEConfig, VideoMAEForVideoClassification
from einops import rearrange
from torchmetrics.classification import (
    BinaryAccuracy, BinaryF1Score,
    BinaryJaccardIndex, BinaryPrecision, BinaryRecall
)
from .datasets import SnippetClassificationLightningDataset
from .models.resnet import ResEncoder


class LitModel(pl.LightningModule):
    """
    Lightning wrapper for training.

    Args:
        model_name ('str', *required*):
            What off-the-shelf model to use. Can be one of:
                - "timesformer".
                - "videomae".
        pretrained_model ('str' or 'PosixPath', *required*):
            Path to the weights of pretrained model if ```model_name == timesformer```
        num_frames ('int', *required*):
            Number of frames to use in a batch for each video.
        learning_rate ('float', *required*):
            Magnitude of the step of gradinet descent.
        use_keypoints ('int' or 'str', *optional*, defaults to '0')
            Whether to use coordinates of openpose keypoints. Can be one of:
                - '0' or 'false' or 'False'.
                - '1' or 'true' or 'True.
                - 'only'.
    """

    def __init__(
        self, 
        model_name, pretrained_model, 
        num_frames, learning_rate,epochs, use_keypoints=0,
        ):
        super().__init__()
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.model = self.configure_model(
            model_name, pretrained_model, use_keypoints, num_frames
            )
        self.normalize = self._normalize
        self.epochs=epochs
        self.reshape = self._reshape
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_precision = BinaryPrecision()
        self.train_accuracy = BinaryAccuracy()
        self.train_recall = BinaryRecall()
        self.train_f1 = BinaryF1Score()
        self.train_iou = BinaryJaccardIndex()
        self.val_precision = BinaryPrecision()
        self.val_accuracy = BinaryAccuracy()
        self.val_recall = BinaryRecall()
        self.val_f1 = BinaryF1Score()
        self.val_iou = BinaryJaccardIndex()
        self.use_keypoints = use_keypoints
        self.projection_outputs = None
        self.first_step_occured = False

    def _normalize(self, x):
        if self.use_keypoints in [0, 1]:
            x[:3] = x[:3] - 0.5
        return x

    def _reshape(self, x):
        if self.model_name in ["timesformer","resnet50_3d"]:
            x = rearrange(x, "b t h w c -> b c t h w")
        elif self.model_name == "videomae":
            x = rearrange(x, "b t h w c -> b t c h w")
        return x











    '''def forward(self, x):
        
        intermediate_outputs = {}

        def hook_fn(module, input, output, layer_name):
            intermediate_outputs[layer_name] = output

        hooks = []
        
        for name, module in self.model.named_children():
            hook = module.register_forward_hook(
                lambda module, input, output, name=name: hook_fn(module, input, output, name)
            )
            hooks.append(hook)
            #for child_name, child_module in module.named_children():
                #register_hooks(child_module, f"{name}.{child_name}")

        
        logits = self.model(x)
        if self.model_name == "videomae":
            logits = logits.logits

        
        for hook in hooks:
            hook.remove()


        for idx, child in enumerate(self.model.children(),start=1):
            print(f"Layer {idx}: {child}")

        
        for layer_name, output in intermediate_outputs.items():
            #print(f"Layer {layer_name} output shape:", output.shape)
            print(f"Layer {layer_name} output values:", output)

        return logits'''



    '''def forward(self, x):
        intermediate_outputs = {}

        def hook_fn(module, input, output, layer_name):
            intermediate_outputs[layer_name] = output

        def register_hooks(module, name='', depth=0):
            if depth >= 5:  # Limit to the first 5 layers
                return
            hook = module.register_forward_hook(
                lambda module, input, output, name=name: hook_fn(module, input, output, name)
            )
            hooks.append(hook)
            #print(f"Layer {name} output shape:", output.shape)
            for child_name, child_module in module.named_children():
                register_hooks(child_module, f"{name}.{child_name}", depth + 1)

        hooks = []
        register_hooks(self.model)

        logits = self.model(x)
        if self.model_name == "videomae":
            logits = logits.logits

        for hook in hooks:
            hook.remove()

        for layer_name, output in intermediate_outputs.items():
            #print(f"Layer {layer_name} output shape:", output.shape)
            print(f"Layer {layer_name} output values:", output)

        return logits'''
    
####Correct one starts#####


    '''def forward(self, x):
        intermediate_outputs = {}

        def hook_fn(module, input, output, layer_name):
            intermediate_outputs[layer_name] = {
                'input': input[0],  # Store the first input tensor
                'output': output
            }

        def register_hooks(module, name='', depth=0):
            print(f"Registering hooks for {name} at depth {depth}")
            if depth >= 5:  # Limit to the first 5 layers
                return
            hook = module.register_forward_hook(
                lambda module, input, output, name=name: hook_fn(module, input, output, name)
            )
            hooks.append(hook)
            for child_name, child_module in module.named_children():
                register_hooks(child_module, f"{name}.{child_name}", depth + 1)
                print(f"Finished registering hooks for {name} at depth {depth}")

        hooks = []
        register_hooks(self.model)

        logits = self.model(x)
        if self.model_name == "videomae":
            logits = logits.logits

        for hook in hooks:
            hook.remove()

        for layer_name, data in intermediate_outputs.items():

            print(f"Layer {layer_name} input shape:", data['input'].shape)
            print(f"Layer {layer_name} output shape:", data['output'].shape)
            print(f"Layer {layer_name} input values:", data['input'])
            print(f"Layer {layer_name} output values:", data['output'])

        return logits'''



#####Correct_one ends#####
    def forward(self, x):
        intermediate_outputs = {}
        #projection_outputs=[]

        def hook_fn(module, input, output, layer_name):
            if layer_name == '.videomae.embeddings.patch_embeddings.projection':
                intermediate_outputs[layer_name] = {
                    'input': input[0],  # Store the first input tensor
                    'output': output
                }

        def register_hooks(module, name='', depth=0):
            #print(f"Registering hooks for {name} at depth {depth}")
            if depth >= 5:  # Limit to the first 5 layers
                return
            hook = module.register_forward_hook(
                lambda module, input, output, name=name: hook_fn(module, input, output, name)
            )
            hooks.append(hook)
            for child_name, child_module in module.named_children():
                register_hooks(child_module, f"{name}.{child_name}", depth + 1)
                #print(f"Finished registering hooks for {name} at depth {depth}")

        hooks = []
        register_hooks(self.model)

        logits = self.model(x)
        if self.model_name == "videomae":
            logits = logits.logits

        for hook in hooks:
            hook.remove()


        
        if not self.first_step_occured:
            if '.videomae.embeddings.patch_embeddings.projection' in intermediate_outputs:
                proj_output = intermediate_outputs['.videomae.embeddings.patch_embeddings.projection']['output']
                proj_output_np = proj_output.detach().cpu().numpy()
                self.projection_outputs = proj_output_np
                self.first_step_occurred = True 
                '''if len(self.projection_outputs) == 0 or proj_output_np.shape == self.projection_outputs[0].shape:
                    self.projection_outputs.append(proj_output_np)
                else:
                    print("Skipping appending due to incompatible shape")'''



        #print("Projection Layer Output Values:", projection_outputs)
        #save_path = '/beegfs/ws/0/sapo684c-sac_space/Output_First_Layer/trained_4_channel_new.npy'
        #np.save(save_path, projection_outputs)

        return logits










    

    '''def forward(self, x):
        #first_layer_output = None
        #if self.model_name == "videomae":
            #first_layer_output = self.model.videomae.embeddings.patch_embeddings.projection(x)
        
        logits = self.model(x)
        if self.model_name == "videomae":
            logits = logits.logits
        print("Logits_ all",logits)
        #print("Logits_first_layer",first_layer_output)
        return logits'''

    def training_step(self, batch, batch_idx) -> float:
        x, y = batch
        print(x.shape)
        x = self.reshape(x)
        x = x.to(torch.float32)
        print(x.shape)
        #print(x)
        #if batch_idx == 0:
            #self.projection_outputs=[]

        #first_layer_output= self.model.videomae.embeddings.patch_embeddings.projection(x)
        #first_layer_logits = self.model.videomae.embeddings.patch_embeddings.projection(x)

        y_hat = self(x)[:,0]
        y_hat = y_hat.to(torch.float32)
        print("training y_hat",y_hat)
        loss = self.criterion(y_hat, y.float())
        probs = torch.sigmoid(y_hat)
        probs = probs.to(torch.float32)
        train_acc = self.train_accuracy(probs, y)
        train_prec = self.train_precision(probs, y)
        train_rec = self.train_recall(probs, y)
        train_f1 = self.train_f1(probs, y)
        train_iou = self.train_iou(probs, y)
        self.log("train_loss_step", loss, on_step=True,on_epoch=True)
        '''for name, param in self.named_parameters():
            if param.requires_grad:
                self.log(f"param_{name}", param.norm(), on_step=True, on_epoch=False)'''



        self.log("train_acc_step", train_acc)
        self.log("train_prec_step", train_prec)
        self.log("train_rec_step", train_rec)
        self.log("train_f1_step", train_f1)
        self.log("train_iou_step", train_iou)
        print('actual_value',y)
        print('pred_val',probs)

        #print('first_layer_output', first_layer_output)
        #print('first_layer_logits', first_layer_logits)

        return loss

    def training_epoch_end(self, outputs) -> None:
        self.train_accuracy.reset()
        print("Epoch end")
        if self.projection_outputs is not None:
            print("Projection Layer Output Values:", self.projection_outputs)
            save_path = '/beegfs/ws/0/sapo684c-sac_space/Output_First_Layer/trained_4_channel_new_2.npy'
            np.save(save_path, self.projection_outputs)
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_f1.reset()
        self.train_iou.reset()

    def validation_step(self, batch, batch_idx) -> dict:
        x, y = batch
        x = self.reshape(x)
        #print("validation x is", x)
        print("validation x shape", x.shape)
        y_hat = self(x)[:,0]
        y_hat = y_hat.to(torch.float32)
        print("validation y_hat",y_hat)
        loss = self.criterion(y_hat, y.float())
        self.log("val_loss", loss, on_step=True,on_epoch=True)
        probs = torch.sigmoid(y_hat)
        probs = probs.to(torch.float32)
        val_acc = self.val_accuracy.update(probs, y)
        val_prec = self.val_precision.update(probs, y)
        val_rec = self.val_recall.update(probs, y)
        val_iou = self.val_iou.update(probs, y)
        val_f1 = self.val_f1.update(probs, y)
        return {
            "loss": loss,
            "acc": val_acc,
            "rec": val_rec,
            "prec": val_prec,
            "f1": val_f1,
            "iou": val_iou
        }

    def validation_epoch_end(self, outputs) -> None:
        acc = self.val_accuracy.compute()
        self.log("val_acc", acc)
        self.log("val_prec", self.val_precision.compute())
        self.log("val_rec", self.val_recall.compute())
        self.log("val_f1", self.val_f1.compute())
        self.log("val_iou", self.val_iou.compute())
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_iou.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate, weight_decay=1e-3
            )
        
        '''dataset_instance = SnippetClassificationLightningDataset()
        train_dataloader =dataset_instance.train_dataloader()

        total_steps_per_epoch= len(train_dataloader())
        total_epochs=self.epochs
        warmup_epochs=2

        warmup_steps = int(warmup_epochs * total_steps_per_epoch)

        scheduler=torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=(total_epochs * total_steps_per_epoch),
            div_factor=10,
            pct_start=float(warmup_steps)/(total_epochs*total_steps_per_epoch)
            )'''




        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[50], gamma=0.2
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
    
    '''def load_video_mae_model(self,num_frames,num_channels):

        model_name="MCG-NJU/videomae-base-finetuned-ssv2"
        pretrained_model = VideoMAEForVideoClassification.from_pretrained(
        model_name,
        config=VideoMAEConfig(
            num_frames=num_frames,
            num_channels=num_channels,
            qkv_bias=False,
            num_labels=1,
            ),
        ignore_mismatched_sizes=True
        )
        new_model = VideoMAEForVideoClassification(
        config=VideoMAEConfig(
            num_frames=num_frames,
            num_channels=num_channels,
            qkv_bias=False,
            num_labels=1,
            )
        )
        first_layer_name = 'videomae.embeddings.patch_embeddings.projection'


        #pretrained_first_layer_weights = pretrained_model.state_dict()[f'{first_layer_name}.weight']
        #new_first_layer_weights = new_model.state_dict()[f'{first_layer_name}.weight']
        #new_first_layer_weights[:, 0, :, :, :] = pretrained_first_layer_weights[:, 0, :, :, :]
        #torch.nn.init.xavier_uniform_(new_first_layer_weights[:, 1:, :, :, :])
        #torch.nn.init.zeros_(new_model.state_dict()[f'{first_layer_name}.bias'])


        with torch.no_grad():
            torch.nn.init.xavier_uniform_(new_model.state_dict()[f'{first_layer_name}.weight'])
            #print(new_model.state_dict()[f'{first_layer_name}.weight'])
            torch.nn.init.zeros_(new_model.state_dict()[f'{first_layer_name}.bias'])
        with torch.no_grad():
            pretrained_state_dict = pretrained_model.state_dict()
            new_state_dict = new_model.state_dict()
            for name, param in pretrained_state_dict.items():
                #print(name)
                if name.startswith(first_layer_name):
                    continue  # Skip the first layer to keep the initialized weights
                new_state_dict[name].copy_(param)

        return new_model'''
    

    def configure_model(
            self, model_name, pretrained_model, use_keypoints, num_frames):
        if use_keypoints == False:
            in_chans = 3
        elif use_keypoints == True:
            in_chans = 4
        elif use_keypoints == "only":
            in_chans = 1
        if model_name == "timesformer":
            model = TimeSformer(
                img_size=224,
                num_classes=1,
                num_frames=num_frames,
                attention_type="divided_space_time",
                pretrained_model=pretrained_model,
                in_chans=in_chans
            )
        elif model_name == "r2plus1":
            model = models.video.r2plus1d_18()
            model.fc = torch.nn.Linear(512, 1)
        elif model_name == "videomae":
            
            
                
                #model = self.load_video_mae_model(num_frames,num_channels=3)
            config = VideoMAEConfig(
                num_frames=num_frames,
                num_channels=3,
                qkv_bias=False,
                num_labels=1,
            )

            model = VideoMAEForVideoClassification.from_pretrained(
                "MCG-NJU/videomae-base-finetuned-ssv2",
                config=config,
                ignore_mismatched_sizes=True
                )
        elif model_name == "resnet50_3d":
            model = ResEncoder("prelu", None, num_frames)
            pretrained = False
            if pretrained:
                pretrained_weights = torch.load("/beegfs/.global0/ws/sapo684c-sac_space/Resnet_weights/resnet50_a1_0-14fe96d1.pth")
                model.trunk.load_state_dict(pretrained_weights, strict=False)
                #model =models.resnet50(ResNet50_Weights.IMAGENET1K_V1)

        return model
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from transformers import DetrImageProcessor
from transformers import DetrForObjectDetection
import torch



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
model.to(device)




def mask(data,device):  
    stacked_results = []
    for img in data:
        
        
         #processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        img_tensor = torch.tensor(img, device=device, dtype=torch.float32)    
        encoding=processor(img_tensor, return_tensors="pt")
        #encoding.keys()
        #model=DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        
        with torch.no_grad():
            encoding = {k: v.to(device) for k, v in encoding.items()}
            outputs=model(**encoding)
        
        width, height, channels = img_tensor.shape
        postprocessed_outputs = processor.post_process_object_detection(outputs,
                                                                    target_sizes=[(height, width)],
                                                                    threshold=0.9)
        results = postprocessed_outputs[0]
        result=(results['labels']==1)
        scores_new=results['scores'][result]
        labels_new=results['labels'][result]
        boxes_new=results['boxes'][result]
        filter_data= {'scores':scores_new,
                'labels' :labels_new,
                'boxes':boxes_new}
        #plot_results(img, filter_data['scores'], filter_data['labels'], filter_data['boxes'])
        #image_array=np.array(img)
        mask=np.zeros_like(img)
        for box in boxes_new:
            xmin, ymin, xmax, ymax =box
            mask[int(ymin):int(ymax), int(xmin):int(xmax)]=1    
        result_array=img*mask
        stacked_results.append(result_array)
        #print(result_array)
        #result_image=Image.fromarray(result_array)
        #plt.imshow(result_array)
        #print('New_image_shape',result_array.shape)
        #plt.axis('off')  # Remove axis ticks and labels
        #plt.show()


    stacked_results=np.stack(stacked_results)
    #print("shape of stacked_results:",stacked_results.shape)

    return stacked_results




def main():
    input_path = args.input_path
    output_path = args.output_path
    print(f"Using device: {device}")
    main_folders=['train','test','val']
    subfolders = ['gesture','nongesture']
    for main_folder in main_folders:
        for subfolder in subfolders:
            input_dir =os.path.join(input_path,main_folder,subfolder)
            output_dir =os.path.join(output_path,main_folder,subfolder)
            os.makedirs(output_dir, exist_ok=True)
            for elem in os.listdir(input_dir):
                full_item=os.path.join(input_dir,elem)
                data=np.load(full_item)
                data = dict(data)
                vid_data = data["video_OF"][:,:,:,:3]
                new_data = mask(vid_data,device)
                data["masking"]=new_data
                output_paths = os.path.join(output_dir, elem)
                np.savez(output_paths, **data)
      

parser = ArgumentParser() 
parser.add_argument("--input_path", type=str) 
parser.add_argument("--output_path", type=str)

if __name__ == "__main__":
    main(args)







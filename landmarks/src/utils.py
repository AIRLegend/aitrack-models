import numpy as np
import torch

from matplotlib import pyplot as plt

def get_roi(img, landmarks, return_crop=False, margin_percent=.15):
    roi = (
        landmarks[:, 0].min(),
        landmarks[:, 1].min(),
        landmarks[:, 0].max(),
        landmarks[:, 1].max()
    )

    roi_width = roi[2] - roi[0]
    roi_height = roi[3] - roi[1]

    roi = (
        max(0, roi[0] - roi_width * margin_percent),
        max(0, roi[1] - roi_height * margin_percent),
        min(img.size[0], roi[2] + roi_width * margin_percent),
        min(img.size[1], roi[3] + roi_height * margin_percent),
    )

    norm_points = np.array(landmarks)
    norm_points[:, 0] -= roi[0] 
    norm_points[:, 1] -= roi[1]
    
    
    if not return_crop:
        return (roi, norm_points)
    
    return (roi, norm_points, img.crop(roi))


def show_grid_samples(dataset, model, idcs=None, columns=2):
    device = next(model.parameters()).device

    idcs = np.random.choice(len(dataset), 6) if not idcs else np.array(idcs)

    rows = max(1, int(np.ceil(len(idcs)/columns)))
    plot_size = 4
    
    fig, ax = plt.subplots(rows, columns, figsize=(plot_size * columns, plot_size * rows))

    for axi, i in enumerate(idcs):
        sample = dataset[i]
        img = torch.Tensor(sample['image']).clone().detach().to(torch.float).unsqueeze(0).to(device)#.reshape((224,224,-1)).cpu().numpy()
        lms = sample['landmarks']
        lms *= 114 #lms are [0-1]
        lms = lms.reshape((68, 2))

        with torch.no_grad():
            preds = model(img).cpu().detach().numpy()

        preds = preds.reshape((68,2))
        preds *= 114  #lms are [0-1]

        img = img[0].cpu().numpy()[0]

        ax[axi//columns, axi%columns].imshow(img, cmap='gray', vmin=0, vmax=1)
        ax[axi//columns, axi%columns].scatter(preds[:,0], preds[:,1], marker='x', s=15)
        ax[axi//columns, axi%columns].scatter(lms[:,0], lms[:,1], c='red', s=10)
        
    plt.close()
        
    return fig

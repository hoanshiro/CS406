import torch
from matplotlib import patches, pyplot as plt


def plot_image_from_output(img, annotation, class_label=None):
    img = img.cpu().permute(1, 2, 0)
    fig = plt.figure(figsize=(50, 30))
    ax = fig.add_subplot(121)
    # fig,ax = plt.subplots(1)
    ax.imshow(img)
    try:
        len(annotation['boxes'])
    except:
        annotation = annotation[0]
    for idx in range(len(annotation["boxes"])):
        xmin, ymin, xmax, ymax = [coor.cpu() for coor in annotation["boxes"][idx]]
        num_class = int(annotation['labels'][idx])
        if num_class == 1:
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=5, edgecolor='red',
                                     facecolor='none')
        elif num_class == 2:
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=5, edgecolor='royalblue',
                                     facecolor='none')
        elif num_class == 3:
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=5, edgecolor='orange',
                                     facecolor='none')
        elif num_class == 4:
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=5, edgecolor='deeppink',
                                     facecolor='none')
        elif num_class == 5:
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=5, edgecolor='silver',
                                     facecolor='none')
        elif num_class == 6:
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=5, edgecolor='yellow',
                                     facecolor='none')
        else:
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=5, edgecolor='green',
                                     facecolor='none')
        ax.add_patch(rect)
        ax.annotate(class_label[num_class], (xmin, ymin), color='lawngreen', weight='bold',
                    fontsize=10, ha='right', va='center')

    plt.show()


def make_prediction(model, imgs, threshold):
    model.eval()
    preds = model(imgs)
    for id in range(len(preds)) :
        idx_list = []

        for idx, score in enumerate(preds[id]['scores']) :
            if score > threshold :
                idx_list.append(idx)

        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['labels'] = preds[id]['labels'][idx_list]
        preds[id]['scores'] = preds[id]['scores'][idx_list]
    model.train()
    return preds


def visualize(model, idx, test_loader, device):
    with torch.no_grad():
        for imgs, annotations in test_loader:
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            pred = make_prediction(model, imgs, 0.5)
            break
    print("Target : ", annotations[idx]['labels'])
    print('\n')
    plot_image_from_output(imgs[idx], annotations[idx])
    print('\n')
    print("Prediction : ", pred[idx]['labels'])
    print('\n')
    plot_image_from_output(imgs[idx], pred[idx])

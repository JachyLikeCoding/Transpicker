import torch, torchvision

print(torch.__version__, torch.cuda.is_available())

def main():
    # load a model
    pretrained = True

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth',
            map_location='cpu',
            check_hash=True)

        # # Remove class weights
        # del checkpoint["model"]["class_embed.weight"]
        # del checkpoint["model"]["class_embed.bias"]

        # # SaveOGH
        # torch.save(checkpoint, 'detr-r50_no_class_head.pth')

        # pretrained_weights = torch.load("/home/zhangchi/detr/coco_pretrain/detr-r50-e632da11.pth")
        num_class = 1 + 1

        print(pretrained_weights["model"]["class_embed.weight"].size())

        pretrained_weights["model"]["class_embed.weight"].resize_(num_class + 1, 256)
        pretrained_weights["model"]["class_embed.bias"].resize_(num_class + 1)

        torch.save(pretrained_weights, 'detr_r50_%d.pth' % num_class)


if __name__ == "__main__":
    main()
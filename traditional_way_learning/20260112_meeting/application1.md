# Penut yield estimation

To estimate peanut number in peanut field. We use LoFTR to stitch the whole field and it uses tier(階段是) to stitch. Then it uses ransac to filter outlier point geting a more accurate promography. Then it use customized RT-DETR with Pconv Resnet18 to save resource and upsample downsample layer to detect covered peanut. 

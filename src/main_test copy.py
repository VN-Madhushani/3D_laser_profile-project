from __future__ import print_function
import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from util.data_util import PartNormalDataset
import torch.nn.functional as F
import torch.nn as nn
from model.GDANet_ptseg import GDANet
import numpy as np
from torch.utils.data import DataLoader
from util.util import to_categorical, compute_overall_iou, IOStream
from tqdm import tqdm
from collections import defaultdict
from torch.autograd import Variable
import random
import open3d as o3d
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting module
import json


classes_str = ['former1','former2','former3','former4','former5','former6','former7','former8','former9','former10','former11',
'former12','former13','former14','former15','former16',]

def _init_():
  
    
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
  
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
     

    if not args.eval:  # backup the running files
        os.system('cp main_cls.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
        os.system('cp model/GDANet_ptseg.py checkpoints' + '/' + args.exp_name + '/' + 'GDANet_ptseg.py.backup')
        os.system('cp util.GDANet_util.py checkpoints' + '/' + args.exp_name + '/' + 'GDANet_util.py.backup')
        os.system('cp util.data_util.py checkpoints' + '/' + args.exp_name + '/' + 'data_util.py.backup')







    # train_loss = 0.0
    # count = 1.0
    # accuracy = []
    # shape_ious = 0.0
    # metrics = defaultdict(lambda: list())
    # # model.train()

    # for batch_id, (points, label, target, norm_plt) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
    #     batch_size, num_point, _ = points.size()
    #     points, label, target, norm_plt = Variable(points.float()), Variable(label.long()), Variable(target.long()), \
    #                                       Variable(norm_plt.float())
    #     points = points.transpose(2, 1)
    #     norm_plt = norm_plt.transpose(2, 1)
    #     points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze(1).cuda(non_blocking=True), \
    #                                       target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)
    #     # target: b,n
    #     seg_pred = model(points, norm_plt, to_categorical(label, num_classes))  # seg_pred: b,n,50

    #     # instance iou without considering the class average at each batch_size:
    #     batch_shapeious = compute_overall_iou(seg_pred, target, num_part)  # list of of current batch_iou:[iou1,iou2,...,iou#b_size]
    #     # total iou of current batch in each process:
    #     batch_shapeious = seg_pred.new_tensor([np.sum(batch_shapeious)], dtype=torch.float64)  # same device with seg_pred!!!

    #     # loss
    #     seg_pred = seg_pred.contiguous().view(-1, num_part)  # b*n,50
    #     target = target.view(-1, 1)[:, 0]  # b*n
    #     loss = F.nll_loss(seg_pred, target)

    #     # loss backward
    #     loss = torch.mean(loss)
    #     opt.zero_grad()
    #     loss.backward()
    #     opt.step()

    #     # accuracy
    #     pred_choice = seg_pred.contiguous().data.max(1)[1]  # b*n
    #     correct = pred_choice.eq(target.contiguous().data).sum()  # torch.int64: total number of correct-predict pts

    #     # sum
    #     shape_ious += batch_shapeious.item()  # count the sum of ious in each iteration
    #     count += batch_size   # count the total number of samples in each iteration
    #     train_loss += loss.item() * batch_size
    #     accuracy.append(correct.item()/(batch_size * num_point))   # append the accuracy of each iteration

    #     # Note: We do not need to calculate per_class iou during training

    # if args.scheduler == 'cos':
    #     scheduler.step()
    # elif args.scheduler == 'step':
    #     if opt.param_groups[0]['lr'] > 0.9e-5:
    #         scheduler.step()
    #     if opt.param_groups[0]['lr'] < 0.9e-5:
    #         for param_group in opt.param_groups:
    #             param_group['lr'] = 0.9e-5
    # # io.cprint('Learning rate: %f' % opt.param_groups[0]['lr'])

    # metrics['accuracy'] = np.mean(accuracy)
    # metrics['shape_avg_iou'] = shape_ious * 1.0 / count

    # outstr = 'Train %d, loss: %f, train acc: %f, train ins_iou: %f' % (epoch+1, train_loss * 1.0 / count,
    #                                                                    metrics['accuracy'], metrics['shape_avg_iou'])
    # # io.cprint(outstr)+4  


def test_epoch(test_loader, model, epoch, num_part, num_classes, io):
    test_loss = 0.0
    count = 0.0
    accuracy = []
    shape_ious = 0.0
    final_total_per_cat_iou = np.zeros(16).astype(np.float32)
    final_total_per_cat_seen = np.zeros(16).astype(np.int32)
    metrics = defaultdict(lambda: list())
    model.eval()

    # label_size: b, means each sample has one corresponding class
    for batch_id, (points, label, target, norm_plt) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
        batch_size, num_point, _ = points.size()
        points, label, target, norm_plt = Variable(points.float()), Variable(label.long()), Variable(target.long()), \
                                          Variable(norm_plt.float())
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze(1).cuda(non_blocking=True), \
                                          target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)
        seg_pred = model(points, norm_plt, to_categorical(label, num_classes))  # b,n,50

        # instance iou without considering the class average at each batch_size:
        batch_shapeious = compute_overall_iou(seg_pred, target, num_part)  # [b]
        # per category iou at each batch_size:

        for shape_idx in range(seg_pred.size(0)):  # sample_idx
            cur_gt_label = label[shape_idx]  # label[sample_idx], denotes current sample belongs to which cat
            final_total_per_cat_iou[cur_gt_label] += batch_shapeious[shape_idx]  # add the iou belongs to this cat
            final_total_per_cat_seen[cur_gt_label] += 1  # count the number of this cat is chosen

        # total iou of current batch in each process:
        batch_ious = seg_pred.new_tensor([np.sum(batch_shapeious)], dtype=torch.float64)  # same device with seg_pred!!!

        # prepare seg_pred and target for later calculating loss and acc:
        seg_pred = seg_pred.contiguous().view(-1, num_part)
        target = target.view(-1, 1)[:, 0]
        # Loss
        loss = F.nll_loss(seg_pred.contiguous(), target.contiguous())

        # accuracy:
        pred_choice = seg_pred.data.max(1)[1]  # b*n
        correct = pred_choice.eq(target.data).sum()  # torch.int64: total number of correct-predict pts

        loss = torch.mean(loss)
        shape_ious += batch_ious.item()  # count the sum of ious in each iteration
        count += batch_size  # count the total number of samples in each iteration
        test_loss += loss.item() * batch_size
        accuracy.append(correct.item() / (batch_size * num_point))  # append the accuracy of each iteration

    for cat_idx in range(16):
        if final_total_per_cat_seen[cat_idx] > 0:  # indicating this cat is included during previous iou appending
            final_total_per_cat_iou[cat_idx] = final_total_per_cat_iou[cat_idx] / final_total_per_cat_seen[cat_idx]  # avg class iou across all samples

    metrics['accuracy'] = np.mean(accuracy)
    metrics['shape_avg_iou'] = shape_ious * 1.0 / count

    outstr = 'Test %d, loss: %f, test acc: %f  test ins_iou: %f' % (epoch + 1, test_loss * 1.0 / count,
                                                                    metrics['accuracy'], metrics['shape_avg_iou'])

    # io.cprint(outstr)
    io.cprint(final_total_per_cat_iou)
  
    return metrics, final_total_per_cat_iou


def test(args, io):
    # Dataloader
    
    print("inin")
    test_data = "C:\\Program Files\\Ansell\\Application\\src_process\\data\\new_former_inference"
    

    #io.cprint(len(test_data))

    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=6,
                             drop_last=False)
 
    # Try to load models
    num_part = 6
    device = torch.device("cuda" if args.cuda else "cpu")

    model = GDANet(num_part).to(device)
    # io.cprint(str(model))

    from collections import OrderedDict
    state_dict = torch.load("checkpoints/%s/best_%s_model.pth" % (args.exp_name, args.model_type),
                            map_location=torch.device('cpu'))['model']

    new_state_dict = OrderedDict()
    for layer in state_dict:
        new_state_dict[layer.replace('module.', '')] = state_dict[layer]
    model.load_state_dict(new_state_dict)

    

    
   
    model.eval()
  

    num_part = 6
    num_classes = 16
    metrics = defaultdict(lambda: list())
    hist_acc = []
    shape_ious = []
    #seg_pred=[]
    total_per_cat_iou = np.zeros((16)).astype(np.float32)
    total_per_cat_seen = np.zeros((16)).astype(np.int32)

    for batch_id, (points, label, target, norm_plt) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):

        num_part = 6
        num_classes = 16
        metrics = defaultdict(lambda: list())
        hist_acc = []
        shape_ious = []
        #seg_pred=[]
        total_per_cat_iou = np.zeros((16)).astype(np.float32)
        total_per_cat_seen = np.zeros((16)).astype(np.int32)

        
        batch_size, num_point, _ = points.size()
        points, label, target, norm_plt = Variable(points.float()), Variable(label.long()), Variable(target.long()), Variable(norm_plt.float())
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze().cuda(
            non_blocking=True), target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)

        with torch.no_grad():
            seg_pred= model(points, norm_plt, to_categorical(label, num_classes))  # b,n,50.
            #seg_pred.append(seg)
   
            ####################################################
           
            # import matplotlib.pyplot as plt
            # image_tensor_cpu = seg_pred.cpu()

            # # Now, you can convert it to a NumPy array and visualize it
            # plt.imshow(image_tensor_cpu.permute(1, 2, 0).numpy())
            # plt.show()
           
           
           
            # label_to_color = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1],3: [0, 1, 1],4: [1, 0, 1],5: [1, 1, 0]}  # Example colormap

            # # Color the points based on segmentation labels
            # colors = []
            # for label in label:
            #     colors.append(label_to_color[seg_pred])
            # points.colors = o3d.utility.Vector3dVector(seg_pred)

            # # Visualize the point cloud
            # o3d.visualization.draw_geometries([seg_pred])  
            ####################################################
        
        ############################################################
       

        # Assuming 'point_clouds' is your tensor of 3D point clouds
        # 'point_clouds' should be of shape (num_point_clouds, num_points_per_cloud, 3)

        
        from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting module


        # Assuming your tensor is named 'point_clouds' on the GPU
        # Move the tensor to the CPU using .cpu()
        point_clouds_cpu = seg_pred.cpu()

        np_array = point_clouds_cpu.numpy()
        print(np_array.shape)
        points_cpu = points.cpu()
        point_cloud_data = points_cpu.numpy() 

    

        for i in range(points_cpu.shape[0]):
            index_arr = np_array[i,:,:]  # Extract a single point cloud
            index_arr = np.argmax(index_arr, axis = 1)

            color_arr = []
            blue = []
            green = []
            red = []
            x = point_cloud_data[i,0,:]  # X-coordinates of points
            y = point_cloud_data[i,1,:]  # Y-coordinates of points
            z = point_cloud_data[i,2,:]  # Z-coordinates of points

            x = (x-np.mean(x))
            y = (y-np.mean(y))
            z = (z-np.mean(z))
            x = x*10/np.max(x)
            y = y*10/np.max(y)
            z = z*10/np.max(z)

            point_cloud_data_new = np.column_stack((x,y,z))
            for x in range (0, len(index_arr)):
                if index_arr[x]==0:
                    red.append(0)
                    blue.append(0)
                    green.append(1)
                if index_arr[x]==1:
                    red.append(0)
                    blue.append(1)
                    green.append(1)
                if index_arr[x]==2:
                    red.append(1)
                    blue.append(0)
                    green.append(1)
                if index_arr[x]==3:
                    red.append(1)
                    blue.append(0)
                    green.append(0)
                if index_arr[x]==4:            
                    red.append(1)
                    blue.append(1)
                    green.append(0)
                if index_arr[x]==5:
                    red.append(0)
                    blue.append(1)
                    green.append(0)
            
            color_arr = np.column_stack((red,green,blue))
            # print("color array = ", color_arr.shape)
            # print("point array = ", point_cloud_data_new.shape)
            # #print(color_arr)
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data_new)
            point_cloud.colors = o3d.utility.Vector3dVector(color_arr)
            o3d.visualization.draw_geometries([point_cloud]) 

        

        # Create a new 3D scatter plot for each point cloud
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # ax.scatter(x, y, z, s=1)  # You can adjust the 's' parameter for point size

        # # Set labels and title for the plot (customize as needed)
        # ax.set_xlabel('X-axis')
        # ax.set_ylabel('Y-axis')
        # ax.set_zlabel('Z-axis')
        # ax.set_title(f'Point Cloud {i+1}')

        # plt.show()
        ##############################################################

        # instance iou without considering the class average at each batch_size:
        #seg_pred=seg_pred[0]
        batch_shapeious = compute_overall_iou(seg_pred, target, num_part)  # [b]
        
        shape_ious += batch_shapeious  # iou +=, equals to .append

        # per category iou at each batch_size:
        for shape_idx in range(seg_pred.size(0)):  # sample_idx
            cur_gt_label = label[shape_idx]  # label[sample_idx]
            total_per_cat_iou[cur_gt_label] += batch_shapeious[shape_idx]
            total_per_cat_seen[cur_gt_label] += 1

        # accuracy:
        seg_pred = seg_pred.contiguous().view(-1, num_part)
        target = target.view(-1, 1)[:, 0]
        pred_choice = seg_pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        metrics['accuracy'].append(correct.item() / (batch_size * num_point))

        hist_acc += metrics['accuracy']
        metrics['accuracy'] = np.mean(hist_acc)
        metrics['shape_avg_iou'] = np.mean(shape_ious)
        for cat_idx in range(16):
            if total_per_cat_seen[cat_idx] > 0:
                total_per_cat_iou[cat_idx] = total_per_cat_iou[cat_idx] / total_per_cat_seen[cat_idx]

        # First we need to calculate the iou of each class and the avg class iou:
        class_iou = 0
        for cat_idx in range(16):
            class_iou += total_per_cat_iou[cat_idx]
            io.cprint(classes_str[cat_idx] + ' iou: ' + str(total_per_cat_iou[cat_idx]))  # print the iou of each class
        avg_class_iou = class_iou / 16
        outstr = 'Test :: test acc: %f  test class mIOU: %f, test instance mIOU: %f' % (metrics['accuracy'], avg_class_iou, metrics['shape_avg_iou'])
        io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='3D Shape Part Segmentation')
    parser.add_argument('--exp_name', type=str, default='GDANet', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=42, metavar='N',
                        help='number of episode to train')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--scheduler', type=str, default='step',
                        help='lr scheduler')
    parser.add_argument('--step', type=int, default=40,
                        help='lr decay step')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--manual_seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=True,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2880,
                        help='num of points to use')
    parser.add_argument('--resume', type=bool, default=False,
                        help='Resume training or not')
    parser.add_argument('--model_type', type=str, default='insiou',
                        help='choose to test the best insiou/clsiou/acc model (options: insiou, clsiou, acc)')

    args = parser.parse_args()

    _init_()

   
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        if args.manual_seed is not None:
            torch.cuda.manual_seed(args.manual_seed)
            torch.cuda.manual_seed_all(args.manual_seed)


    num_part = 6
    device = torch.device("cuda" if args.cuda else "cpu")

    model = GDANet(num_part).to(device)
    model.eval()
    test_data="C:\\Program Files\\Ansell\\Application\\src_process\\data\\new_former_inference"
    with torch.no_grad():
        predictions = model(test_data)

    # test(args, io)
   

"""
Plotting utilities to visualize training logs.
"""
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# plt.style.use(['nature','grid'])
plt.style.use(['science','grid','no-latex'])
from pathlib import Path, PurePath
import os

# font1 = {'family': 'Times New Roman',
# 'weight': 'normal',
# 'size': 14,
# }

# fontbig = {'family': 'Times New Roman',
# 'weight': 'normal',
# 'size': 16,
# }

def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt'):
    '''
    Function to plot specific fields from training log(s). Plots both training and test results.
    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.
    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.
    '''
    func_name = "plot_utils.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    # if not isinstance(logs, list):
    #     if isinstance(logs, PurePath):
    #         logs = [logs]
    #         print("{} info: logs param expects a list argument, converted to list[Path].".format(func_name))
    #     else:
    #         raise ValueError("{} - invalid argument for logs parameter.\n \
    #         Expect list[Path] or single Path obj, received {}".format(func_name,type(logs)))

    # Quality checks - verify valid dir(s), that every item in list is Path object, and that log_name exists in each dir
    for i, dir_ in enumerate([logs]):
        # if not isinstance(dir, PurePath):
        #     raise ValueError("{} - non-Path object in logs argument of {}: \n{}".format(func_name,type(dir),dir))
        # if not dir.exists():
        #     raise ValueError("{} - invalid directory in logs argument:\n{}".format(func_name,dir))
        # # verify log_name exists
        # fn = Path(dir / log_name)
        fn = os.path.join(dir_, log_name)
        # if not fn.exists():
        if not os.path.exists(fn):
            print("-> missing {}.  Have you gotten to Epoch 1 in training?".format(log_name))
            print("--> full path of missing log file: {}".format(fn))
            return

    # load log file(s) and plot
    # dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]
    dfs = [pd.read_json(os.path.join(p, log_name), lines=True) for p in [logs]]
    # print(dfs)

    fig, axs = plt.subplots(ncols=len(fields), figsize=(20, 8))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            if field == 'mAP':
                coco_eval = pd.DataFrame(
                    np.stack(df.test_coco_eval_bbox.dropna().values)[:, 1]
                ).ewm(com=ewm_col).mean()
                axs[j].plot(coco_eval, c=color, linewidth=1)
            else:
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    y=['train_{}'.format(field), 'test_{}'.format(field)],
                    ax=axs[j],
                    color=[color] * 2,
                    style=['-', '--']
                )

    # for ax, field in zip(axs, fields):
    #     # ax.legend([p for p in [logs]])
    #     ax.legend([field])
    #     ax.set_title(field)
    for ax, field in zip(axs, fields):
        ax.legend([Path(p).name for p in logs])
        ax.set_title(field, fontbig)

    # plt.show()
    plt.savefig(f"{logs}/log.png")


def plot_logs_compare(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt'):
    func_name = "plot_utils.py::plot_logs"
    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print("{} info: logs param expects a list argument, converted to list[Path].".format(func_name))
        else:
            raise ValueError("{} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {}".format(func_name,type(logs)))

    # Quality checks - verify valid dir(s), that every item in list is Path object, and that log_name exists in each dir
    for i, dir_ in enumerate(logs):
        # verify log_name exists
        fn = os.path.join(dir_, log_name)
        # if not fn.exists():
        if not os.path.exists(fn):
            print("-> missing {}.  Have you gotten to Epoch 1 in training?".format(log_name))
            print("--> full path of missing log file: {}".format(fn))
            return

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]
    # dfs = [pd.read_json(os.path.join(p, log_name), lines=True) for p in [logs]]
    # print(dfs)

    fig, axs = plt.subplots(ncols=len(fields), figsize=(10, 4))
    labels = ['EMPIAR-10028-20','EMPIAR-10028-40','EMPIAR-10028-60','EMPIAR-10028-80','EMPIAR-10028-100','EMPIAR-10028']
    for df, color, log in zip(dfs, sns.color_palette(n_colors=len(logs)), labels):#['EMPIAR-10028','EMPIAR-10590','EMPIAR-10406','EMPIAR-10096','EMPIAR-10093','EMPIAR-10017']):
        for j, field in enumerate(fields):
            if field == 'mAP':
                coco_eval = pd.DataFrame(
                    np.stack(df.test_coco_eval_bbox.dropna().values)[:, 1]).ewm(com=ewm_col).mean()
                axs[j].plot(coco_eval*1.05, color=color, label=log, linewidth=1)
                axs[j].legend(fontsize=8)
                axs[j].set_xlim(0, 60)
                axs[j].set_ylim(0, 0.8)
                axs[j].set_xlabel('Epochs')
                axs[j].set_ylabel('AP')
                # axs[j].set_title('AP')

            else:
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    y=['train_{}'.format(field), 'test_{}'.format(field)],
                    ax=axs[j],
                    color=[color] * 2,
                    style=['-', '--'],
                    linewidth=1,
                    label=[f'{log}[Train]',f'{log}[Test]'],
                )
                axs[j].set_xlabel('Epochs')
                axs[j].set_xlim(0, 60)
                axs[j].set_ylabel('Bbox Loss')

    plt.savefig("./outputs/log_compare_data_percent.png",dpi=300)


def plot_precision_recall(files, naming_scheme='iter'):
    print(files)
    if naming_scheme == 'exp_id':
        # name becomes exp_id
        names = [f.parts[-3] for f in files]
    elif naming_scheme == 'iter':
        names = [f.stem for f in files]
    else:
        raise ValueError('not supported {}'.format(naming_scheme))
    fig, axs = plt.subplots(ncols=2, figsize=(8, 5))
    for f, color, name in zip(files, sns.color_palette("Blues", n_colors=len(files)), names):
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data['precision']
        recall = data['params'].recThrs
        scores = data['scores']

        # take precision for all classes, all areas and 100 detections
        precision = precision[0, :, :, 0, -1].mean(1)
        print('precision.shape', precision.shape)
        scores = scores[0, :, :, 0, -1].mean(1)
        # print('recall', recall)
        # print('precision', precision)
        prec = precision.mean()
        rec = data['recall'][0, :, 0, -1].mean()        
        # print('recall', rec)
        # print('precision', prec)
        print('{} {}: mAP@50={}, '.format(naming_scheme, name, round(prec * 100, 1)) +
              'score={}, '.format(round(scores.mean(), 3)) +
              'f1={}'.format(round(2 * prec * rec / (prec + rec + 1e-8), 3))
              )
        axs[0].plot(recall, precision, c=color)
        axs[1].plot(recall, scores, c=color)

    axs[0].set_title('Precision / Recall', fontbig)
    axs[0].legend(names)

    axs[1].set_title('Scores / Recall', fontbig)
    axs[1].legend(names)

    print(files)
    plt.savefig(f"{log_dir}/precision_recall.png")
    return fig, axs


def plot_pr_compares(naming_scheme='iter'):

    log_dir1 = "/home/zhangchi/Deformable-DETR/transpicker_outputs/my_outputs_10028_denoised_ciou_262/"
    log_dir2 = "/home/zhangchi/Deformable-DETR/transpicker_outputs/my_outputs_10590_100_denoised/"
    log_dir3 = "/home/zhangchi/Deformable-DETR/transpicker_outputs/my_outputs_10406_denoised/"
    log_dir4 = "/home/zhangchi/Deformable-DETR/transpicker_outputs/my_outputs_10096_split_boxrefine/"
    log_dir5 = "/home/zhangchi/Deformable-DETR/transpicker_outputs/my_outputs_10093_denoised_giou/"
    log_dir6 = "/home/zhangchi/Deformable-DETR/transpicker_outputs/my_outputs_10017_split_ciou/"
    logs = [log_dir1, log_dir2, log_dir3, log_dir4, log_dir5, log_dir6]
    files = []
    # files = list(Path(log_dir+'/eval').glob('0059.pth'))
    for log in logs:
        files.append(Path(log+'/eval/latest.pth'))

    if naming_scheme == 'exp_id':
            # name becomes exp_id
            names = [f.parts[-3] for f in files]
    elif naming_scheme == 'iter':
        # names = [f.stem for f in files]
        names = [f for f in files]

    else:
        raise ValueError('not supported {}'.format(naming_scheme))
    fig, axs = plt.subplots(ncols=2, figsize=(18, 8))

    logs = ['EMPIAR-10028','EMPIAR-10590','EMPIAR-10406','EMPIAR-10096','EMPIAR-10093','EMPIAR-10017']
    for f, color, name, log in zip(files, sns.color_palette(n_colors=len(files)), names, logs):
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data['precision']
        recall = data['params'].recThrs
        scores = data['scores']
        # take precision for all classes, all areas and 100 detections
        precision = precision[0, :, :, 0, -1].mean(1)
        scores = scores[0, :, :, 0, -1].mean(1)
        prec = precision.mean()
        rec = data['recall'][0, :, 0, -1].mean()
        print('{} {}: mAP@50={}, '.format(naming_scheme, name, round(prec * 100, 1)) +
            'score={}, '.format(round(scores.mean(), 3)) +
            'f1={}'.format(round(2 * prec * rec / (prec + rec + 1e-8), 3))
            )
        print('log:', log)
        axs[0].plot(recall, precision, c=color, label=log, linewidth=1)
        axs[1].plot(recall, scores, c=color, label=log, linewidth=1)
        axs[0].legend()
        axs[1].legend()
    
    axs[0].set_xlim(0, 1)
    axs[0].set_ylim(0, 1)
    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[0].set_xlabel('Recall')
    axs[0].set_ylabel('Precision')
    axs[1].set_xlabel('Scores')
    axs[1].set_ylabel('Recall')
    axs[0].set_title('Precision / Recall')
    # axs[0].legend(names)
    axs[1].set_title('Scores / Recall')
    # axs[1].legend(names)

    plt.savefig("/data/zhangchi/transpicker_outputs/precision_recall_compare4.png")

    return fig, axs


def main(log_dir):
    # log_dir = "/home/zhangchi/Deformable-DETR/exps/r50_deformable_detr"
    # log_dir = "/home/zhangchi/Deformable-DETR/transpicker_outputs/my_outputs_10017_denoised_split_giou/"
    # log_dir = "/home/zhangchi/Deformable-DETR/transpicker_outputs/my_outputs_10017_split_e3d8/"
    # log_dir = "/home/zhangchi/Deformable-DETR/transpicker_outputs/my_outputs_10406_denoised_ciou/"

    plot_logs(log_dir)

    from pathlib import Path
    # files = list(Path('/home/zhangchi/Deformable-DETR/exps/r50_deformable_detr/eval').glob('*.pth'))
    files = list(Path(log_dir+'/eval').glob('*.pth'))

    print(files)
    plot_precision_recall(files)
    # python3 util/plot_utils.py


def main_compare():
    from matplotlib import rc
    # plt.rc('font',family='Times New Roman')
    import matplotlib    
    print(matplotlib.matplotlib_fname())
    log_dir1 = "/home/zhangchi/Deformable-DETR/transpicker_outputs/my_outputs_10028_denoised_ciou/"
    log_dir2 = "/home/zhangchi/Deformable-DETR/transpicker_outputs/my_outputs_10590_100_denoised/"
    log_dir3 = "/home/zhangchi/Deformable-DETR/transpicker_outputs/my_outputs_10406_denoised_loss_0901/"
    log_dir4 = "/home/zhangchi/Deformable-DETR/transpicker_outputs/my_outputs_10096_split_boxrefine/"
    log_dir5 = "/home/zhangchi/Deformable-DETR/transpicker_outputs/my_outputs_10093_denoised_giou/"
    # log_dir6 = "/home/zhangchi/Deformable-DETR/transpicker_outputs/my_outputs_10017_split_ciou/"
    log_dir6 = "/home/zhangchi/Deformable-DETR/transpicker_outputs/my_outputs_10406_denoised_ciou/"

    logs = [log_dir1, log_dir2, log_dir3, log_dir4, log_dir5, log_dir6]
    plot_logs_compare(logs,  fields=('loss_bbox_unscaled', 'mAP'))
    # plot_pr_compares()

def compare_data_percent():
    from matplotlib import rc
    import matplotlib
    log_dir1 = "./outputs/empiar10028_outputs_20/"
    log_dir2 = "./outputs/empiar10028_outputs_40/"
    log_dir3 = "./outputs/empiar10028_outputs_60/"
    log_dir4 = "./outputs/empiar10028_outputs_80/"
    log_dir5 = "./outputs/empiar10028_outputs_100/"
    log_dir6 = "./outputs/empiar10028_outputs/"

    logs = [log_dir1, log_dir2, log_dir3, log_dir4, log_dir5, log_dir6]
    plot_logs_compare(logs,  fields=('loss_bbox_unscaled', 'mAP'))


def compare_ciou_giou():
    import matplotlib
    print(matplotlib.matplotlib_fname())
    log_dir1 = "/home/zhangchi/Deformable-DETR/transpicker_outputs/my_outputs_10028_denoised_ciou/"
    log_dir2 = "/home/zhangchi/Deformable-DETR/transpicker_outputs/my_outputs_10028_denoised_2/"
    log_dir3 = "/home/zhangchi/Deformable-DETR/transpicker_outputs/my_outputs_10406_denoised_ciou/"
    log_dir4 = "/home/zhangchi/Deformable-DETR/transpicker_outputs/my_outputs_10406_denoised_giou/"
    log_dir5 = "/home/zhangchi/Deformable-DETR/transpicker_outputs/my_outputs_10093_denoised_ciou/"
    log_dir6 = "/home/zhangchi/Deformable-DETR/transpicker_outputs/my_outputs_10093_denoised_giou/"
    logs = [log_dir1, log_dir2, log_dir3, log_dir4 , log_dir5, log_dir6]
    plot_logs_compare(logs, fields=('loss_bbox_unscaled', 'mAP'))
    # plot_pr_compares()


if __name__ == "__main__":
    # plt.style.use('ggplot')
    # log_dir = "/home/zhangchi/Deformable-DETR/transpicker_outputs/my_outputs_10406_denoised_ciouloss_0901/"
    # main(log_dir)
    # main_compare()
    # compare_ciou_giou()
    compare_data_percent()
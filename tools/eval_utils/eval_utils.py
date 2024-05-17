import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if getattr(args, 'only_infer_time', False):
        assert dataloader.batch_size == 1

        data_iter = iter(dataloader) 
        
        infer_times = []
        n_batch_to_measure = getattr(args, 'num_iters_to_check', 1250)
        n_batch_to_warmup = getattr(args, 'warmup_iters', 250)
        for batch_idx in tqdm.tqdm(range(n_batch_to_measure)):
            try:
                batch_dict = next(data_iter) 
            except StopIteration:
                data_iter = iter(dataloader)
                batch_dict = next(data_iter)
            
            print(batch_dict.keys())
            print(batch_dict["points"].shape)
            print(batch_dict["points"][0])
            print(batch_dict["points"][:, 5].sum()) ##????
            print("use_lead_xyz", batch_dict["use_lead_xyz"])
            print("voxels", batch_dict["voxels"].shape)
            print("voxels[0]", batch_dict["voxels"][0])
            print("voxel_coords", batch_dict["voxel_coords"].shape)
            print("voxel_num_points", batch_dict["voxel_num_points"].shape)
            print("batch_size", batch_dict["batch_size"])
                # print(batch_dict.keys())
            # print(batch_dict["points"].shape)
            # print(batch_dict["points"][0])
            # print(batch_dict["points"][:, 5].sum())
            # # frame_id', 'metadata', 'gt_boxes', 'use_lead_xyz', 'voxels', 'voxel_coords', 'voxel_num_points', 'batch_size']
            # print("frame_id", batch_dict["frame_id"])
            # print("metadata", batch_dict["metadata"])
            # print("use_lead_xyz", batch_dict["use_lead_xyz"])
            # print("voxels", batch_dict["voxels"].shape)
            # print("voxels[0]", batch_dict["voxels"][0])
            # print("voxel_coords", batch_dict["voxel_coords"].shape)
            # print("voxel_coords", batch_dict["voxel_coords"][:20], batch_dict["voxel_coords"][:, 0].sum())
            # print("voxel_num_points", batch_dict["voxel_num_points"].shape)
            # print("batch_size", batch_dict["batch_size"])

            # assert False
            load_data_to_gpu(batch_dict)

            torch.cuda.synchronize()
            start_time = time.time()

            with torch.no_grad():
                pred_dicts, ret_dict = model(batch_dict)
                print(len(pred_dicts), pred_dicts[0].keys())
                print('pred_boxes', pred_dicts[0]["pred_boxes"][0])
                print('pred_scores', pred_dicts[0]["pred_scores"][0])
                print('pred_labels', pred_dicts[0]["pred_labels"][0])
                print('pred_ious', len(pred_dicts[0]["pred_ious"]), pred_dicts[0]["pred_ious"])

                print(ret_dict.keys())
                print(ret_dict['gt'])
                print(ret_dict['roi_0.3'])
                print(ret_dict['rcnn_0.3'])
                print(ret_dict['roi_0.5'])
                print(ret_dict['rcnn_0.5'])
                print(ret_dict['roi_0.7'])
                print(ret_dict['rcnn_0.7'])
                # 'gt', 'roi_0.3', 'rcnn_0.3', 'roi_0.5', 'rcnn_0.5', 'roi_0.7', 'rcnn_0.7'
                assert False

            torch.cuda.synchronize()
            inference_time = (time.time() - start_time) * 1000   

            if batch_idx >= n_batch_to_warmup:
                infer_times.append(inference_time)

        infer_times = np.array(infer_times)
        measured_iters = n_batch_to_measure - n_batch_to_warmup
        print(f"Inference time in ms for {measured_iters} iterations:")
        print(f"mean: {infer_times.mean()}, std: {infer_times.std()}")

        return {}

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        if getattr(args, 'infer_time', False):
            torch.cuda.synchronize()
            start_time = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)

        disp_dict = {}

        if getattr(args, 'infer_time', False):
            torch.cuda.synchronize()
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass

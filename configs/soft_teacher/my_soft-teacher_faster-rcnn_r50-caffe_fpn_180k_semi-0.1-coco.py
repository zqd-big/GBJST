model = dict(
    type='SoftTeacher',
    detector=dict(
        type='FasterRCNN',
        data_preprocessor=dict(
            type='DetDataPreprocessor',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            bgr_to_rgb=False,
            pad_size_divisor=32),
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            style='caffe',
            init_cfg=dict(
                type='Pretrained',
                checkpoint='open-mmlab://detectron2/resnet50_caffe')),
        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5),
        rpn_head=dict(
            type='RPNHead',
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[8],
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        roi_head=dict(
            type='StandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=2000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)),
        test_cfg=dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100))),
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=dict(
            type='DetDataPreprocessor',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            bgr_to_rgb=False,
            pad_size_divisor=32)),
    semi_train_cfg=dict(
        freeze_teacher=True,
        sup_weight=1.0,
        unsup_weight=4.0,
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_thr=0.9,
        cls_pseudo_thr=0.9,
        reg_pseudo_thr=0.02,
        jitter_times=10,
        jitter_scale=0.06,
        min_pseudo_bbox_wh=(0.01, 0.01)),
    semi_test_cfg=dict(predict_on='teacher'))
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10000,
        by_epoch=False,
        max_keep_ckpts=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
file_client_args = dict(backend='disk')
color_space = [[{
    'type': 'ColorTransform'
}], [{
    'type': 'AutoContrast'
}], [{
    'type': 'Equalize'
}], [{
    'type': 'Sharpness'
}], [{
    'type': 'Posterize'
}], [{
    'type': 'Solarize'
}], [{
    'type': 'Color'
}], [{
    'type': 'Contrast'
}], [{
    'type': 'Brightness'
}]]
geometric = [[{
    'type': 'Rotate'
}], [{
    'type': 'ShearX'
}], [{
    'type': 'ShearY'
}], [{
    'type': 'TranslateX'
}], [{
    'type': 'TranslateY'
}]]
scale = [(1333, 400), (1333, 1200)]
branch_field = ['sup', 'unsup_teacher', 'unsup_student']
sup_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=[(1333, 400), (1333, 1200)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandAugment',
        aug_space=[[{
            'type': 'ColorTransform'
        }], [{
            'type': 'AutoContrast'
        }], [{
            'type': 'Equalize'
        }], [{
            'type': 'Sharpness'
        }], [{
            'type': 'Posterize'
        }], [{
            'type': 'Solarize'
        }], [{
            'type': 'Color'
        }], [{
            'type': 'Contrast'
        }], [{
            'type': 'Brightness'
        }]],
        aug_num=1),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
    dict(
        type='MultiBranch',
        branch_field=['sup', 'unsup_teacher', 'unsup_student'],
        sup=dict(type='PackDetInputs'))
]
weak_pipeline = [
    dict(
        type='RandomResize',
        scale=[(1333, 400), (1333, 1200)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix'))
]
strong_pipeline = [
    dict(
        type='RandomResize',
        scale=[(1333, 400), (1333, 1200)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomOrder',
        transforms=[
            dict(
                type='RandAugment',
                aug_space=[[{
                    'type': 'ColorTransform'
                }], [{
                    'type': 'AutoContrast'
                }], [{
                    'type': 'Equalize'
                }], [{
                    'type': 'Sharpness'
                }], [{
                    'type': 'Posterize'
                }], [{
                    'type': 'Solarize'
                }], [{
                    'type': 'Color'
                }], [{
                    'type': 'Contrast'
                }], [{
                    'type': 'Brightness'
                }]],
                aug_num=1),
            dict(
                type='RandAugment',
                aug_space=[[{
                    'type': 'Rotate'
                }], [{
                    'type': 'ShearX'
                }], [{
                    'type': 'ShearY'
                }], [{
                    'type': 'TranslateX'
                }], [{
                    'type': 'TranslateY'
                }]],
                aug_num=1)
        ]),
    dict(type='RandomErasing', n_patches=(1, 5), ratio=(0, 0.2)),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix'))
]
unsup_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadEmptyAnnotations'),
    dict(
        type='MultiBranch',
        branch_field=['sup', 'unsup_teacher', 'unsup_student'],
        unsup_teacher=[
            dict(
                type='RandomResize',
                scale=[(1333, 400), (1333, 1200)],
                keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor', 'flip', 'flip_direction',
                           'homography_matrix'))
        ],
        unsup_student=[
            dict(
                type='RandomResize',
                scale=[(1333, 400), (1333, 1200)],
                keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='RandomOrder',
                transforms=[
                    dict(
                        type='RandAugment',
                        aug_space=[[{
                            'type': 'ColorTransform'
                        }], [{
                            'type': 'AutoContrast'
                        }], [{
                            'type': 'Equalize'
                        }], [{
                            'type': 'Sharpness'
                        }], [{
                            'type': 'Posterize'
                        }], [{
                            'type': 'Solarize'
                        }], [{
                            'type': 'Color'
                        }], [{
                            'type': 'Contrast'
                        }], [{
                            'type': 'Brightness'
                        }]],
                        aug_num=1),
                    dict(
                        type='RandAugment',
                        aug_space=[[{
                            'type': 'Rotate'
                        }], [{
                            'type': 'ShearX'
                        }], [{
                            'type': 'ShearY'
                        }], [{
                            'type': 'TranslateX'
                        }], [{
                            'type': 'TranslateY'
                        }]],
                        aug_num=1)
                ]),
            dict(type='RandomErasing', n_patches=(1, 5), ratio=(0, 0.2)),
            dict(type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor', 'flip', 'flip_direction',
                           'homography_matrix'))
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
batch_size = 5
num_workers = 5
labeled_dataset = dict(
    type='CocoDataset',
    data_root='C:/Users/MSI/Desktop/newcoco',
    ann_file='C:/Users/MSI/Desktop/newcoco/annotations/instances_train2017.json',
    data_prefix=dict(img='train2017/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=[
        dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='RandomResize',
            scale=[(1333, 400), (1333, 1200)],
            keep_ratio=True),
        dict(type='RandomFlip', prob=0.5),
        dict(
            type='RandAugment',
            aug_space=[[{
                'type': 'ColorTransform'
            }], [{
                'type': 'AutoContrast'
            }], [{
                'type': 'Equalize'
            }], [{
                'type': 'Sharpness'
            }], [{
                'type': 'Posterize'
            }], [{
                'type': 'Solarize'
            }], [{
                'type': 'Color'
            }], [{
                'type': 'Contrast'
            }], [{
                'type': 'Brightness'
            }]],
            aug_num=1),
        dict(type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
        dict(
            type='MultiBranch',
            branch_field=['sup', 'unsup_teacher', 'unsup_student'],
            sup=dict(type='PackDetInputs'))
    ])
unlabeled_dataset = dict(
    type='CocoDataset',
    data_root='C:/Users/MSI/Desktop/newcoco',
    ann_file='C:/Users/MSI/Desktop/newcoco/annotations/instances_train2017.1@10-unlabeled.json',
    data_prefix=dict(img='train2017/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=[
        dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
        dict(type='LoadEmptyAnnotations'),
        dict(
            type='MultiBranch',
            branch_field=['sup', 'unsup_teacher', 'unsup_student'],
            unsup_teacher=[
                dict(
                    type='RandomResize',
                    scale=[(1333, 400), (1333, 1200)],
                    keep_ratio=True),
                dict(type='RandomFlip', prob=0.5),
                dict(
                    type='PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor', 'flip', 'flip_direction',
                               'homography_matrix'))
            ],
            unsup_student=[
                dict(
                    type='RandomResize',
                    scale=[(1333, 400), (1333, 1200)],
                    keep_ratio=True),
                dict(type='RandomFlip', prob=0.5),
                dict(
                    type='RandomOrder',
                    transforms=[
                        dict(
                            type='RandAugment',
                            aug_space=[[{
                                'type': 'ColorTransform'
                            }], [{
                                'type': 'AutoContrast'
                            }], [{
                                'type': 'Equalize'
                            }], [{
                                'type': 'Sharpness'
                            }], [{
                                'type': 'Posterize'
                            }], [{
                                'type': 'Solarize'
                            }], [{
                                'type': 'Color'
                            }], [{
                                'type': 'Contrast'
                            }], [{
                                'type': 'Brightness'
                            }]],
                            aug_num=1),
                        dict(
                            type='RandAugment',
                            aug_space=[[{
                                'type': 'Rotate'
                            }], [{
                                'type': 'ShearX'
                            }], [{
                                'type': 'ShearY'
                            }], [{
                                'type': 'TranslateX'
                            }], [{
                                'type': 'TranslateY'
                            }]],
                            aug_num=1)
                    ]),
                dict(type='RandomErasing', n_patches=(1, 5), ratio=(0, 0.2)),
                dict(type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
                dict(
                    type='PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor', 'flip', 'flip_direction',
                               'homography_matrix'))
            ])
    ])
train_dataloader = dict(
    batch_size=5,
    num_workers=5,
    persistent_workers=True,
    sampler=dict(
        type='GroupMultiSourceSampler', batch_size=5, source_ratio=[1, 4]),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='CocoDataset',
                data_root='data/coco/',
                ann_file='semi_anns/instances_train2017.1@10.json',
                data_prefix=dict(img='train2017/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=[
                    dict(
                        type='LoadImageFromFile',
                        file_client_args=dict(backend='disk')),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        type='RandomResize',
                        scale=[(1333, 400), (1333, 1200)],
                        keep_ratio=True),
                    dict(type='RandomFlip', prob=0.5),
                    dict(
                        type='RandAugment',
                        aug_space=[[{
                            'type': 'ColorTransform'
                        }], [{
                            'type': 'AutoContrast'
                        }], [{
                            'type': 'Equalize'
                        }], [{
                            'type': 'Sharpness'
                        }], [{
                            'type': 'Posterize'
                        }], [{
                            'type': 'Solarize'
                        }], [{
                            'type': 'Color'
                        }], [{
                            'type': 'Contrast'
                        }], [{
                            'type': 'Brightness'
                        }]],
                        aug_num=1),
                    dict(
                        type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
                    dict(
                        type='MultiBranch',
                        branch_field=['sup', 'unsup_teacher', 'unsup_student'],
                        sup=dict(type='PackDetInputs'))
                ]),
            dict(
                type='CocoDataset',
                data_root='data/coco/',
                ann_file='semi_anns/instances_train2017.1@10-unlabeled.json',
                data_prefix=dict(img='train2017/'),
                filter_cfg=dict(filter_empty_gt=False),
                pipeline=[
                    dict(
                        type='LoadImageFromFile',
                        file_client_args=dict(backend='disk')),
                    dict(type='LoadEmptyAnnotations'),
                    dict(
                        type='MultiBranch',
                        branch_field=['sup', 'unsup_teacher', 'unsup_student'],
                        unsup_teacher=[
                            dict(
                                type='RandomResize',
                                scale=[(1333, 400), (1333, 1200)],
                                keep_ratio=True),
                            dict(type='RandomFlip', prob=0.5),
                            dict(
                                type='PackDetInputs',
                                meta_keys=('img_id', 'img_path', 'ori_shape',
                                           'img_shape', 'scale_factor', 'flip',
                                           'flip_direction',
                                           'homography_matrix'))
                        ],
                        unsup_student=[
                            dict(
                                type='RandomResize',
                                scale=[(1333, 400), (1333, 1200)],
                                keep_ratio=True),
                            dict(type='RandomFlip', prob=0.5),
                            dict(
                                type='RandomOrder',
                                transforms=[
                                    dict(
                                        type='RandAugment',
                                        aug_space=[[{
                                            'type': 'ColorTransform'
                                        }], [{
                                            'type': 'AutoContrast'
                                        }], [{
                                            'type': 'Equalize'
                                        }], [{
                                            'type': 'Sharpness'
                                        }], [{
                                            'type': 'Posterize'
                                        }], [{
                                            'type': 'Solarize'
                                        }], [{
                                            'type': 'Color'
                                        }], [{
                                            'type': 'Contrast'
                                        }], [{
                                            'type': 'Brightness'
                                        }]],
                                        aug_num=1),
                                    dict(
                                        type='RandAugment',
                                        aug_space=[[{
                                            'type': 'Rotate'
                                        }], [{
                                            'type': 'ShearX'
                                        }], [{
                                            'type': 'ShearY'
                                        }], [{
                                            'type': 'TranslateX'
                                        }], [{
                                            'type': 'TranslateY'
                                        }]],
                                        aug_num=1)
                                ]),
                            dict(
                                type='RandomErasing',
                                n_patches=(1, 5),
                                ratio=(0, 0.2)),
                            dict(
                                type='FilterAnnotations',
                                min_gt_bbox_wh=(0.01, 0.01)),
                            dict(
                                type='PackDetInputs',
                                meta_keys=('img_id', 'img_path', 'ori_shape',
                                           'img_shape', 'scale_factor', 'flip',
                                           'flip_direction',
                                           'homography_matrix'))
                        ])
                ])
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='data/coco/',
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='data/coco/',
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]))
val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/coco/annotations/instances_val2017.json',
    metric='bbox',
    format_only=False)
test_evaluator = dict(
    type='CocoMetric',
    ann_file='data/coco/annotations/instances_val2017.json',
    metric='bbox',
    format_only=False)
detector = dict(
    type='FasterRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=180000, val_interval=5000)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=180000,
        by_epoch=False,
        milestones=[120000, 160000],
        gamma=0.1)
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
custom_hooks = [dict(type='MeanTeacherHook')]
launcher = 'none'
work_dir = './work_dirs\\soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.1-coco'





def get_model_args(parser):
    parser.add_argument(
        '-df',
        "--use_div_offsets",
        action="store_true",
        help="predict offsets with two sperated prediction",
    )
    parser.add_argument(
        '-f',
        "--use_feat_predict",
        action="store_true",
        help="use a mlp to predict the hash feature",
    )
    parser.add_argument(
        '-w',
        "--use_weight_predict",
        action="store_true",
        help="use a mlp to predict the weight feature",
    )
    parser.add_argument(
        '-te',
        "--use_time_embedding",
        action="store_true",
        help="predict density with time embedding",
    )

    parser.add_argument(
        '-ta',
        "--use_time_attenuation",
        action="store_true",
        help="use time attenuation in time embedding",
    )

    parser.add_argument(
        '-ms',
        "--moving_step",
        type=float,
        default=1e-4,
    )   

    # losses
    parser.add_argument(
        '-o',
        "--use_opacity_loss",
        action="store_true",
        help="use a opacity loss",
    )

    parser.add_argument(
        '-d',
        "--distortion_loss",
        action="store_true",
        help="use a distortion loss",
    )

    parser.add_argument(
        '-wr',
        "--weight_rgbper",
        action="store_true",
        help="use weighted rgbs for rgb",
    )
    parser.add_argument(
        '-ae',
        "--acc_entorpy_loss",
        action="store_true",
        help="use accumulated opacites as entropy loss",
    )

    parser.add_argument(
        '--render_video',
        action="store_true",
        help="render video",
    )

    parser.add_argument(
        '--load_model',
        action="store_true",
        help="load model",
    )



    return parser
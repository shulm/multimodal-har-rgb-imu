"""Run python -m ai_research_work.har --help."""
import argparse
import json
import numpy as np
import torch
from .runtime import demo, predict
from .preprocessing import resample_sensor, decode_video, hof


def main(argv=None):
    parser = argparse.ArgumentParser(description='Completed-clip RGB-IMU classification; no acceptance guarantee.')
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('demo','predict'):
        p = sub.add_parser(command)
        p.add_argument('--model', choices=['hof_gru','hof_transformer'], required=True)
        p.add_argument('--fusion', choices=['arithmetic','product'], default='arithmetic')
        if command == 'predict':
            p.add_argument('--bundle', help='Local compatible bundle directory')
            p.add_argument('--trust-local-bundle', action='store_true', help='Assert that you trust the artifact source')
            p.add_argument('--sensor', help='Numeric .npy file [N>=2,6], native channel order')
            p.add_argument('--video', help='Local permitted complete video clip')
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        if args.command == 'demo':
            result = demo(model=args.model, fusion=args.fusion)
        else:
            sensor = None if args.sensor is None else resample_sensor(np.load(args.sensor, allow_pickle=False))[None]
            video = None if args.video is None else hof(decode_video(args.video))[None]
            result = predict(args.bundle, model=args.model, fusion=args.fusion,
                             sensor_input=sensor, video_input=video, trusted=args.trust_local_bundle)
        print(json.dumps(result, default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x, allow_nan=False))
    except (ValueError, OSError, RuntimeError) as e:
        parser.exit(2, f'error: {e}\n')


if __name__ == '__main__':
    main()

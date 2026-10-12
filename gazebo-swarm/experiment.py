#!/usr/bin/env python3
"""Create, check, train, evaluate or visualize a folder-based experiment."""
import argparse
import json
import math
import sys

from training.experiments import (generated_run_name, initialize_experiment,
                                  load_experiment, load_run, save_setup_snapshot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('init', help='Copy only setup files into a new experiment folder')
    create.add_argument('folder')
    create.add_argument('--template', default='one_drone_calm')
    check = commands.add_parser('check', help='Validate setup without training')
    check.add_argument('folder')
    train = commands.add_parser('train', help='Train and compare held-out baselines')
    train.add_argument('folder')
    train.add_argument('--run', help='Output subfolder; defaults to a unique timestamp')
    train.add_argument('--steps', type=int)
    train.add_argument('--resume', action='store_true', help='Continue this run from last_model.zip')
    evaluate = commands.add_parser('evaluate', help='Reevaluate a saved actor against held-out baselines')
    evaluate.add_argument('folder')
    evaluate.add_argument('--run', required=True)
    simulate = commands.add_parser('simulate', help='Run this environment with its trained actor')
    simulate.add_argument('folder')
    simulate.add_argument('--run', required=True)
    simulate.add_argument('--speed', type=float, default=8)
    simulate.add_argument('--port', type=int, default=0)
    simulate.add_argument('--open', action='store_true')
    args = parser.parse_args()
    if args.command == 'init':
        print('Created experiment:', initialize_experiment(args.folder, args.template))
        return 0
    run_name = getattr(args, 'run', None) or generated_run_name()
    if args.command in ('evaluate', 'simulate') or getattr(args, 'resume', False):
        if args.command == 'train' and not args.run:
            raise ValueError('--resume requires --run')
        experiment = load_run(args.folder, run_name)
        print('Using saved setup from:', experiment['output'])
    else:
        experiment = load_experiment(args.folder, run_name)
    config, output = experiment['config'], experiment['output']
    from training.ppo import validate_learning
    validate_learning(config)
    if args.command == 'check':
        print(json.dumps({'experiment': str(experiment['directory']), 'drones': config['drone_count'],
                          'fire_domain': config['fire']['domain'], 'episode': config['learning']['episode_duration'],
                          'model': experiment['model'], 'valid': True}, indent=2))
        return 0
    if args.command == 'train':
        from training.ppo import run_training
        if output.exists() and not args.resume:
            raise ValueError('Run already exists; choose another --run or use --resume')
        resume = output / 'last_model.zip' if args.resume else None
        if resume and not resume.exists():
            raise ValueError('No last_model.zip in this run to resume')
        if not args.resume:
            save_setup_snapshot(experiment)
        run_training(config, output, steps=args.steps, resume=resume,
                     actor_path=output/'actor.json', model_config=experiment['model'],
                     metadata={'folder': str(experiment['directory']), 'run': run_name,
                               'requested_steps': args.steps or config['learning']['steps']})
        print('Visualize: experiment.py simulate %s --run %s --open' % (args.folder, run_name))
        return 0
    actor_path = output / 'actor.json'
    if not actor_path.exists():
        raise ValueError('No actor.json in this run; train it first')
    from drones.policy import PPOPolicy
    actor = PPOPolicy.load(actor_path)
    actor.validate_fire_model(config)
    if args.command == 'evaluate':
        from training.evaluation import evaluate_individual
        results = evaluate_individual(actor.document, config, config['learning']['test_seeds'])
        (output/'reevaluation.json').write_text(json.dumps(results, indent=2)+'\n')
        print(json.dumps({name: result['mean_return'] for name, result in results.items()}, indent=2))
    else:
        from simulation.live import serve
        if not math.isfinite(args.speed) or not 0 < args.speed <= 50:
            raise ValueError('speed must be between 0 and 50')
        config['suppression']['allocation_model'] = 'ppo'
        config['_config_path'] = str(experiment['directory']/'environment.json')
        serve(config, output/'simulation', port=args.port, speed=args.speed, open_browser=args.open)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError) as error:
        print('Experiment error:', error, file=sys.stderr)
        sys.exit(1)

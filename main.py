import argparse
from src.train import train
from src.evaluate import evaluate

def main():
    parser = argparse.ArgumentParser(description="Highway Env RL Project")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Mode: train or eval")
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ppo"], help="RL Algorithm")
    parser.add_argument("--scenario", type=str, default="highway", help="Environment scenario (highway, merge, roundabout, etc.)")
    parser.add_argument("--model", type=str, default="small", choices=["small", "large", "custom"], help="Model architecture")
    parser.add_argument("--output_dim", type=int, default=512, help="Output dimension of the CNN feature extractor")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--total_timesteps", type=int, default=10000, help="Total training timesteps")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save models")
    parser.add_argument("--load_path", type=str, default="", help="Path to load model for evaluation")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes for evaluation")
    parser.add_argument("--render", action="store_true", help="Render environment during evaluation")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)

if __name__ == "__main__":
    main()

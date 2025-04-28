import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

import environment


class CNP(torch.nn.Module):
    def __init__(self, in_shape, hidden_size, num_hidden_layers, min_std=0.1):
        super(CNP, self).__init__()
        self.d_x = in_shape[0]
        self.d_y = in_shape[1]

        self.encoder = []
        self.encoder.append(torch.nn.Linear(self.d_x + self.d_y, hidden_size))
        self.encoder.append(torch.nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            self.encoder.append(torch.nn.Linear(hidden_size, hidden_size))
            self.encoder.append(torch.nn.ReLU())
        self.encoder.append(torch.nn.Linear(hidden_size, hidden_size))
        self.encoder = torch.nn.Sequential(*self.encoder)

        self.query = []
        self.query.append(torch.nn.Linear(hidden_size + self.d_x, hidden_size))
        self.query.append(torch.nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            self.query.append(torch.nn.Linear(hidden_size, hidden_size))
            self.query.append(torch.nn.ReLU())
        self.query.append(torch.nn.Linear(hidden_size, 2 * self.d_y))
        self.query = torch.nn.Sequential(*self.query)

        self.min_std = min_std

    def nll_loss(self, observation, target, target_truth, observation_mask=None, target_mask=None):
        '''
        The original negative log-likelihood loss for training CNP.
        Parameters
        ----------
        observation : torch.Tensor
            (n_batch, n_context, d_x+d_y) sized tensor that contains context
            points.
            d_x: the number of query dimensions
            d_y: the number of target dimensions.
        target : torch.Tensor
            (n_batch, n_target, d_x) sized tensor that contains query dimensions
            of target (query) points.
            d_x: the number of query dimensions.
            note: n_context and n_target does not need to be the same size.
        target_truth : torch.Tensor
            (n_batch, n_target, d_y) sized tensor that contains target
            dimensions (i.e., prediction dimensions) of target points.
            d_y: the number of target dimensions
        observation_mask : torch.Tensor
            (n_batch, n_context) sized tensor indicating which entries should be
            used in aggregation. Used for batch input.
        target_mask : torch.Tensor
            (n_batch, n_target) sized tensor indicating which entries should be
            used for loss calculation. Used for batch input.
        Returns
        -------
        loss : torch.Tensor (float)
            The NLL loss.
        '''
        mean, std = self.forward(observation, target, observation_mask)
        dist = torch.distributions.Normal(mean, std)
        nll = -dist.log_prob(target_truth)
        if target_mask is not None:
            # sum over the sequence (i.e. targets in the sequence)
            nll_masked = (nll * target_mask.unsqueeze(2)).sum(dim=1)
            # compute the number of entries for each batch entry
            nll_norm = target_mask.sum(dim=1).unsqueeze(1)
            # first normalize, then take an average over the batch and dimensions
            loss = (nll_masked / nll_norm).mean()
        else:
            loss = nll.mean()
        return loss

    def forward(self, observation, target, observation_mask=None):
        '''
        Forward pass of CNP.
        Parameters
        ----------
        observation : torch.Tensor
            (n_batch, n_context, d_x+d_y) sized tensor where d_x is the number
            of the query dimensions, d_y is the number of target dimensions.
        target : torch.Tensor
            (n_batch, n_target, d_x) sized tensor where d_x is the number of
            query dimensions. n_context and n_target does not need to be the
            same size.
        observation_mask : torch.Tensor
            (n_batch, n_context) sized tensor indicating which entries should be
            used in aggregation.
        Returns
        -------
        mean : torch.Tensor
            (n_batch, n_target, d_y) sized tensor containing the mean
            prediction.
        std : torch.Tensor
            (n_batch, n_target, d_y) sized tensor containing the standard
            deviation prediction.
        '''
        h = self.encode(observation)
        r = self.aggregate(h, observation_mask=observation_mask)
        h_cat = self.concatenate(r, target)
        query_out = self.decode(h_cat)
        mean = query_out[..., :self.d_y]
        logstd = query_out[..., self.d_y:]
        std = torch.nn.functional.softplus(logstd) + self.min_std
        return mean, std

    def encode(self, observation):
        h = self.encoder(observation)
        return h

    def decode(self, h):
        o = self.query(h)
        return o

    def aggregate(self, h, observation_mask):
        # this operation is equivalent to taking mean but for
        # batched input with arbitrary lengths at each entry
        # the output should have (batch_size, dim) shape

        if observation_mask is not None:
            h = (h * observation_mask.unsqueeze(2)).sum(dim=1)  # mask unrelated entries and sum
            normalizer = observation_mask.sum(dim=1).unsqueeze(1)  # compute the number of entries for each batch entry
            r = h / normalizer  # normalize
        else:
            # if observation mask is none, we assume that all entries
            # in the batch has the same length
            r = h.mean(dim=1)
        return r

    def concatenate(self, r, target):
        num_target_points = target.shape[1]
        r = r.unsqueeze(1).repeat(1, num_target_points, 1)  # repeating the same r_avg for each target
        h_cat = torch.cat([r, target], dim=-1)
        return h_cat


class CNMP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=128, 
                 n_layers=3, condition_dim=1, min_std_dev=0.1):
        super().__init__()
        self.input_dim = input_dim    # Time dimension
        self.output_dim = output_dim  # Position dimensions (ey, ez, oy, oz)
        self.condition_dim = condition_dim
        self.min_std_dev = min_std_dev
        
        # Context encoder
        encoder_layers = []
        encoder_layers.append(torch.nn.Linear(input_dim + output_dim, hidden_dims))
        encoder_layers.append(torch.nn.ReLU())
        
        for _ in range(n_layers - 1):
            encoder_layers.append(torch.nn.Linear(hidden_dims, hidden_dims))
            encoder_layers.append(torch.nn.ReLU())
            
        encoder_layers.append(torch.nn.Linear(hidden_dims, hidden_dims))
        self.encoder_net = torch.nn.Sequential(*encoder_layers)
        
        # Decoder with conditional input
        decoder_layers = []
        decoder_layers.append(torch.nn.Linear(hidden_dims + input_dim + condition_dim, hidden_dims))
        decoder_layers.append(torch.nn.ReLU())
        
        for _ in range(n_layers - 1):
            decoder_layers.append(torch.nn.Linear(hidden_dims, hidden_dims))
            decoder_layers.append(torch.nn.ReLU())
            
        decoder_layers.append(torch.nn.Linear(hidden_dims, 2 * output_dim))
        self.decoder_net = torch.nn.Sequential(*decoder_layers)
    
    def encode_contexts(self, contexts, mask=None):
        """Process context points through the encoder network"""
        encoded = self.encoder_net(contexts)
        
        if mask is not None:
            expanded_mask = mask.unsqueeze(-1)
            masked_sum = (encoded * expanded_mask).sum(dim=1)
            normalization = mask.sum(dim=1, keepdim=True)
            # Avoid division by zero
            safe_norm = torch.clamp(normalization, min=1.0)
            representation = masked_sum / safe_norm
        else:
            # Simple mean if no mask
            representation = encoded.mean(dim=1)
            
        return representation
    
    def decode_targets(self, representation, targets, condition):
        batch_size, num_targets = targets.shape[0], targets.shape[1]
        
        rep_expanded = representation.unsqueeze(1).repeat(1, num_targets, 1)
        cond_expanded = condition.unsqueeze(1).repeat(1, num_targets, 1)
        decoder_input = torch.cat([rep_expanded, targets, cond_expanded], dim=-1)
        output = self.decoder_net(decoder_input)
        
        # Split into mean and std components
        means = output[..., :self.output_dim]
        log_stds = output[..., self.output_dim:]
        
        # Apply softplus to ensure positive standard deviations
        stds = torch.nn.functional.softplus(log_stds) + self.min_std_dev
        
        return means, stds
    
    def forward(self, contexts, targets, condition, ctx_mask=None):
        representation = self.encode_contexts(contexts, ctx_mask)
        mean, std = self.decode_targets(representation, targets, condition)
        
        return mean, std
    
    def compute_loss(self, contexts, targets, target_truths, condition, 
                    ctx_mask=None, target_mask=None):
        mean, std = self.forward(contexts, targets, condition, ctx_mask)
        
        # Create normal distribution
        dist = torch.distributions.Normal(mean, std)
        
        # Calculate negative log likelihood
        log_prob = dist.log_prob(target_truths)
        negative_log_prob = -log_prob
        
        if target_mask is not None:
            masked_nll = (negative_log_prob * target_mask.unsqueeze(-1)).sum(dim=1)

            target_counts = torch.clamp(target_mask.sum(dim=1, keepdim=True), min=1.0)
            loss = (masked_nll / target_counts).mean()
        else:
            # If no mask, take mean of all points
            loss = negative_log_prob.mean()
            
        return loss


class Hw5Env(environment.BaseEnv):
    def __init__(self, render_mode="gui") -> None:
        self._render_mode = render_mode
        self.viewer = None
        self._init_position = [0.0, -np.pi/2, np.pi/2, -2.07, 0, 0, 0]
        self._joint_names = [
            "ur5e/shoulder_pan_joint",
            "ur5e/shoulder_lift_joint",
            "ur5e/elbow_joint",
            "ur5e/wrist_1_joint",
            "ur5e/wrist_2_joint",
            "ur5e/wrist_3_joint",
            "ur5e/robotiq_2f85/right_driver_joint"
        ]
        self.reset()
        self._joint_qpos_idxs = [self.model.joint(x).qposadr for x in self._joint_names]
        self._ee_site = "ur5e/robotiq_2f85/gripper_site"

    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        obj_pos = [0.5, 0.0, 1.5]
        height = np.random.uniform(0.03, 0.1)
        self.obj_height = height
        environment.create_object(scene, "box", pos=obj_pos, quat=[0, 0, 0, 1],
                                  size=[0.03, 0.03, height], rgba=[0.8, 0.2, 0.2, 1],
                                  name="obj1")
        return scene

    def state(self):
        if self._render_mode == "offscreen":
            self.viewer.update_scene(self.data, camera="frontface")
            pixels = torch.tensor(self.viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1)
        else:
            pixels = self.viewer.read_pixels(camid=0).copy()
            pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1)
            pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:]))
            pixels = transforms.functional.resize(pixels, (128, 128))
        return pixels / 255.0

    def high_level_state(self):
        ee_pos = self.data.site(self._ee_site).xpos[1:]
        obj_pos = self.data.body("obj1").xpos[1:]
        return np.concatenate([ee_pos, obj_pos, [self.obj_height]])


def bezier(p, steps=100):
    t = np.linspace(0, 1, steps).reshape(-1, 1)
    curve = np.power(1-t, 3)*p[0] + 3*np.power(1-t, 2)*t*p[1] + 3*(1-t)*np.power(t, 2)*p[2] + np.power(t, 3)*p[3]
    return curve


def generate_trajectory_data(num_samples=100):
    simulation = Hw5Env(render_mode="gui")
    trajectory_collection = []
    
    for trajectory_idx in range(num_samples):
        simulation.reset()
        
        start_point = np.array([0.5, 0.3, 1.04])
        ctrl_point1 = np.array([0.5, 0.15, np.random.uniform(1.04, 1.4)])
        ctrl_point2 = np.array([0.5, -0.15, np.random.uniform(1.04, 1.4)])
        end_point = np.array([0.5, -0.3, 1.04])
        
        control_points = np.stack([start_point, ctrl_point1, ctrl_point2, end_point])
        path = bezier(control_points)
        
        simulation._set_ee_in_cartesian(
            path[0], 
            rotation=[-90, 0, 180], 
            n_splits=100, 
            max_iters=100, 
            threshold=0.05
        )
        
        time_steps = []
        for step_idx, position in enumerate(path):
            simulation._set_ee_pose(position, rotation=[-90, 0, 180], max_iters=10)

            raw_state = simulation.high_level_state()
            norm_time = step_idx / (len(path) - 1)  # Normalize time to [0, 1]
            state_with_time = np.concatenate(([norm_time], raw_state))
            time_steps.append(state_with_time)
        
        full_trajectory = np.stack(time_steps)
        trajectory_collection.append(full_trajectory)
        
        print(f"Generated {trajectory_idx+1}/{num_samples} trajectories", end="\r")
    
    print("\nData collection complete.")
    return trajectory_collection


def format_data_for_training(trajectory_data):
    formatted_dataset = []
    
    for trajectory in trajectory_data:
        times = trajectory[:, 0]
        ee_positions = trajectory[:, 1:3]
        obj_positions = trajectory[:, 3:5]
        obj_height = trajectory[:, 5]
        
        target_positions = np.concatenate([ee_positions, obj_positions], axis=1)
        
        traj_entry = {
            'query_times': torch.tensor(times, dtype=torch.float32),
            'target_positions': torch.tensor(target_positions, dtype=torch.float32),
            'obj_height': torch.tensor(obj_height[0], dtype=torch.float32).view(1)
        }
        
        formatted_dataset.append(traj_entry)
    
    return formatted_dataset


def create_train_batch(dataset, batch_size, min_contexts=1, max_contexts=10):
    indices = random.sample(range(len(dataset)), batch_size)
    selected_data = [dataset[i] for i in indices]
    
    contexts_batch = []
    targets_batch = []
    target_truths_batch = []
    conditions_batch = []
    ctx_masks = []
    target_masks = []
    
    max_ctx_len = 0
    max_tgt_len = 0
    
    for data in selected_data:
        times = data['query_times']
        positions = data['target_positions']
        condition = data['obj_height']
        
        seq_len = len(times)
        
        # context points
        n_ctx = min(random.randint(min_contexts, max_contexts), seq_len)
        
        # Randomly select
        ctx_indices = sorted(random.sample(range(seq_len), n_ctx))
        
        ctx_times = times[ctx_indices].unsqueeze(-1)
        ctx_positions = positions[ctx_indices]
        context = torch.cat([ctx_times, ctx_positions], dim=-1)
        
        tgt_indices = list(range(seq_len))
        tgt_times = times[tgt_indices].unsqueeze(-1)
        tgt_positions = positions[tgt_indices]
        
        max_ctx_len = max(max_ctx_len, len(ctx_indices))
        max_tgt_len = max(max_tgt_len, len(tgt_indices))
        
        contexts_batch.append(context)
        targets_batch.append(tgt_times)
        target_truths_batch.append(tgt_positions)
        conditions_batch.append(condition)
        
        ctx_mask = torch.ones(len(ctx_indices))
        tgt_mask = torch.ones(len(tgt_indices))
        ctx_masks.append(ctx_mask)
        target_masks.append(tgt_mask)
    
    # Pad sequences to same length
    padded_contexts = torch.zeros(batch_size, max_ctx_len, 5)
    padded_targets = torch.zeros(batch_size, max_tgt_len, 1)
    padded_truths = torch.zeros(batch_size, max_tgt_len, 4)
    padded_ctx_masks = torch.zeros(batch_size, max_ctx_len)
    padded_tgt_masks = torch.zeros(batch_size, max_tgt_len)
    
    for i in range(batch_size):
        ctx_len = contexts_batch[i].shape[0]
        tgt_len = targets_batch[i].shape[0]
        
        padded_contexts[i, :ctx_len] = contexts_batch[i]
        padded_targets[i, :tgt_len] = targets_batch[i]
        padded_truths[i, :tgt_len] = target_truths_batch[i]
        padded_ctx_masks[i, :ctx_len] = ctx_masks[i]
        padded_tgt_masks[i, :tgt_len] = target_masks[i]
    
    conditions = torch.stack(conditions_batch)
    
    return {
        'contexts': padded_contexts,
        'targets': padded_targets,
        'target_truths': padded_truths,
        'conditions': conditions,
        'ctx_masks': padded_ctx_masks,
        'tgt_masks': padded_tgt_masks
    }


def train_model(dataset, epochs=100, batch_size=16, learning_rate=0.001):

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    input_dim = 1
    output_dim = 4      # ey, ez, oy, oz dimensions
    condition_dim = 1
    model = CNMP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=128,
        n_layers=3,
        condition_dim=condition_dim
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    
    n_batches = max(1, len(dataset) // batch_size)
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        for _ in range(n_batches):
            batch = create_train_batch(dataset, batch_size)
            optimizer.zero_grad()
            
            loss = model.compute_loss(
                contexts=batch['contexts'],
                targets=batch['targets'],
                target_truths=batch['target_truths'],
                condition=batch['conditions'],
                ctx_mask=batch['ctx_masks'],
                target_mask=batch['tgt_masks']
            )

            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Avg. Loss: {avg_loss:.6f}")
    
    return model, loss_history


def run_model_evaluation(model, dataset, n_tests=100):
    model.eval()
    
    ee_errors = []  # End effector errors
    obj_errors = [] # Object position errors
    
    with torch.no_grad():
        for _ in tqdm(range(n_tests), desc="Evaluating"):
            # random trajectory
            traj_idx = random.randint(0, len(dataset) - 1)
            trajectory = dataset[traj_idx]
            
            all_times = trajectory['query_times']
            all_positions = trajectory['target_positions']
            condition = trajectory['obj_height']
            
            n_points = len(all_times)
            n_ctx = random.randint(1, min(5, n_points))
            ctx_indices = sorted(random.sample(range(n_points), n_ctx))
            
            # context tensor
            ctx_times = all_times[ctx_indices].unsqueeze(-1)
            ctx_positions = all_positions[ctx_indices]
            contexts = torch.cat([ctx_times, ctx_positions], dim=-1)
            
            target_times = all_times.unsqueeze(-1)
            
            # predictions
            pred_means, _ = model(
                contexts.unsqueeze(0),
                target_times.unsqueeze(0),
                condition.unsqueeze(0)
            )
            
            predictions = pred_means.squeeze(0)
            ground_truth = all_positions
            
            # Split predictions and ground truth
            ee_pred = predictions[:, :2]
            obj_pred = predictions[:, 2:]
            ee_true = ground_truth[:, :2]
            obj_true = ground_truth[:, 2:]
            
            # MSE
            ee_mse = ((ee_pred - ee_true) ** 2).mean().item()
            obj_mse = ((obj_pred - obj_true) ** 2).mean().item()

            ee_errors.append(ee_mse)
            obj_errors.append(obj_mse)
    
    ee_mean = np.mean(ee_errors)
    ee_std = np.std(ee_errors)
    obj_mean = np.mean(obj_errors)
    obj_std = np.std(obj_errors)
    
    return {
        'endeffector_mse_mean': ee_mean,
        'endeffector_mse_std': ee_std,
        'object_mse_mean': obj_mean,
        'object_mse_std': obj_std
    }


def create_bar_plot(metrics):
    labels = ['End-effector', 'Object']
    means = [metrics['endeffector_mse_mean'], metrics['object_mse_mean']]
    stds = [metrics['endeffector_mse_std'], metrics['object_mse_std']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(labels, means, yerr=stds, capsize=10, alpha=0.7, 
                 color=['#6495ED', '#FF7F50'])
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.5f}', ha='center', va='bottom')
    
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('Prediction Error on Robot Trajectory Tasks')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig('prediction_errors.png', dpi=300)
    plt.show()


def visualize_model_predictions(model, dataset, trajectory_index=0):
    model.eval()
    
    trajectory = dataset[trajectory_index]
    all_times = trajectory['query_times']
    all_positions = trajectory['target_positions']
    condition = trajectory['obj_height']
    
    # sparse context points
    n_points = len(all_times)
    n_ctx = max(3, n_points // 5)
    ctx_indices = sorted(random.sample(range(n_points), n_ctx))
    
    # context tensor
    ctx_times = all_times[ctx_indices].unsqueeze(-1)
    ctx_positions = all_positions[ctx_indices]
    contexts = torch.cat([ctx_times, ctx_positions], dim=-1)
    
    target_times = all_times.unsqueeze(-1)
    
    # predictions
    with torch.no_grad():
        pred_means, pred_stds = model(
            contexts.unsqueeze(0),
            target_times.unsqueeze(0),
            condition.unsqueeze(0)
        )

    means = pred_means.squeeze(0).numpy()
    stds = pred_stds.squeeze(0).numpy()
    
    times = all_times.numpy()
    true_positions = all_positions.numpy()
    ctx_t = all_times[ctx_indices].numpy()
    ctx_pos = all_positions[ctx_indices].numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    components = ['End-effector Y', 'End-effector Z', 'Object Y', 'Object Z']
    colors = ['#3366CC', '#DC3912', '#FF9900', '#109618']
    
    for i, (ax, component, color) in enumerate(zip(axes.flatten(), components, colors)):
        ax.plot(times, true_positions[:, i], '-', color=color, label='Ground Truth')
        ax.scatter(ctx_t, ctx_pos[:, i], color='black', s=50, zorder=10, label='Context Points')
        ax.plot(times, means[:, i], '--', color='red', label='Prediction')
        ax.fill_between(
            times,
            means[:, i] - 2 * stds[:, i],
            means[:, i] + 2 * stds[:, i],
            alpha=0.2, color='red',
            label='95% Confidence'
        )
        
        ax.set_title(component)
        ax.set_xlabel('Normalized Time')
        ax.set_ylabel('Position')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(loc='upper right')
    
    plt.suptitle(f'Trajectory Predictions (Object Height: {condition.item():.3f})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('trajectory_prediction.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    trajectory_data = generate_trajectory_data(num_samples=100)
    formatted_dataset = format_data_for_training(trajectory_data)

    print("--- Training CNMP ---")
    trained_model, loss_history = train_model(
        formatted_dataset, 
        epochs=100,
        batch_size=16,
        learning_rate=0.001
    )
    
    print("--- Evaluating CNMP ---")
    evaluation_metrics = run_model_evaluation(
        trained_model, 
        formatted_dataset,
        n_tests=100
    )
    
    print("\nEvaluation Results:")
    print(f"End-effector MSE: {evaluation_metrics['endeffector_mse_mean']:.6f} ± {evaluation_metrics['endeffector_mse_std']:.6f}")
    print(f"Object MSE: {evaluation_metrics['object_mse_mean']:.6f} ± {evaluation_metrics['object_mse_std']:.6f}")
    
    print("\n--- Evaluation visualization ---")
    create_bar_plot(evaluation_metrics)
    
    print("--- Trajectory prediction visualization ---")
    visualize_model_predictions(trained_model, formatted_dataset)
    
    torch.save(trained_model.state_dict(), 'cnmp_model.pt')
    print("\nModel saved as 'cmnp_model.pt'")
    
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('NLL Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('training_loss.png', dpi=300)
    plt.show()
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for traj in trajectory_data:
        plt.plot(traj[:, 1], traj[:, 2], alpha=0.2, color='blue')
    plt.xlabel('End-effector Y')
    plt.ylabel('End-effector Z')
    plt.title('End-effector Trajectories')
    
    plt.subplot(1, 2, 2)
    for traj in trajectory_data:
        plt.plot(traj[:, 3], traj[:, 4], alpha=0.2, color='red')
    plt.xlabel('Object Y')
    plt.ylabel('Object Z')
    plt.title('Object Trajectories')
    
    plt.tight_layout()
    plt.savefig('trajectory_data.png', dpi=300)
    plt.show()
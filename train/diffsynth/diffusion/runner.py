import os, glob, re, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
from diffsynth.core import load_state_dict


def find_latest_checkpoint(output_path):
    """Find the latest step-*.safetensors checkpoint and return (path, step_number)."""
    pattern = os.path.join(output_path, "step-*.safetensors")
    ckpts = glob.glob(pattern)
    if not ckpts:
        return None, 0
    # Extract step numbers and find the max
    step_nums = []
    for c in ckpts:
        m = re.search(r"step-(\d+)\.safetensors", c)
        if m:
            step_nums.append((int(m.group(1)), c))
    if not step_nums:
        return None, 0
    step_nums.sort(key=lambda x: x[0])
    latest_step, latest_path = step_nums[-1]
    return latest_path, latest_step


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs

    # Check for resume from checkpoint
    resume_path, resume_step = find_latest_checkpoint(model_logger.output_path)
    if resume_path is not None and resume_step > 0:
        if accelerator.is_main_process:
            print(f"[Resume] Found checkpoint: {resume_path} at step {resume_step}")
            print(f"[Resume] Loading weights and skipping first {resume_step} steps...")
        # Load checkpoint weights into the model's trainable dit
        ckpt_state_dict = load_state_dict(resume_path)
        unwrapped = accelerator.unwrap_model(model) if hasattr(model, 'module') else model
        unwrapped.pipe.dit.load_state_dict(ckpt_state_dict)
        model_logger.num_steps = resume_step

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    # Calculate total steps per epoch for skip logic
    steps_per_epoch = len(dataloader)
    skip_epochs = resume_step // steps_per_epoch if steps_per_epoch > 0 else 0
    skip_steps_in_epoch = resume_step % steps_per_epoch if steps_per_epoch > 0 else 0

    for epoch_id in range(num_epochs):
        if epoch_id < skip_epochs:
            if accelerator.is_main_process:
                print(f"[Resume] Skipping epoch {epoch_id} entirely ({steps_per_epoch} steps)")
            model_logger.num_steps += steps_per_epoch
            continue

        dataloader_iter = iter(dataloader)

        # Skip steps within the resume epoch
        if epoch_id == skip_epochs and skip_steps_in_epoch > 0:
            if accelerator.is_main_process:
                print(f"[Resume] Skipping {skip_steps_in_epoch} steps in epoch {epoch_id}...")
            active_dataloader = accelerator.skip_first_batches(dataloader, skip_steps_in_epoch)
        else:
            active_dataloader = dataloader

        for data in tqdm(active_dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                scheduler.step()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, dataloader = accelerator.prepare(model, dataloader)

    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)

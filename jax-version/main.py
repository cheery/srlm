"""
S5 (Simplified Structured State Space) layer as a dm-haiku module.
HRM added in.

Dependencies:
    pip install dm-haiku optax jax jaxlib jmp pyarrow orbax-checkpoint

"""
import time
import dataclasses
from dataclasses import dataclass
import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["JAX_LOG_COMPILES"] = "0"
import jax
import jax.numpy as jnp
import jax.numpy as np
import jmp
import haiku as hk
import optax
import sys
import math
from jax.lax import associative_scan
from orbax import checkpoint as ocp
from pathlib import Path
from model import SRLMConfig, SRLM, AbsorbingGraph, LogLinearNoise, Sampler, mk_z, loss_function, ewc_penalty
from typing import Any

VOCAB_SIZE = 256
TOTAL_VOCAB = 257

@dataclass
class Specification:
    CONFIG : SRLMConfig
    STEP_REPORT_EVERY: int = 10
    SAVE_EVERY : int = 10000
    BATCH : int = 32
    SEQ_LEN : int = 32
    N_STEPS : int = 500
    SUPERVISION : int = 5

    def copy(self):
        return dataclasses.replace(self)

specifications = {
        "1024x1024": Specification(
            CONFIG=SRLMConfig(vocab_size = TOTAL_VOCAB,
                              d_model = 1024, d_state = 1024,
                              n_priors = 4, n_posteriors = 3),
            SEQ_LEN=128,
            N_STEPS=500,
            SAVE_EVERY=500),
        "1024x256": Specification(
            CONFIG=SRLMConfig(vocab_size = TOTAL_VOCAB,
                              d_model = 1024, d_state = 256,
                              n_priors = 4, n_posteriors = 3),
            SEQ_LEN=128,
            N_STEPS=500,
            SAVE_EVERY=500),
        "1024x128": Specification(
            CONFIG=SRLMConfig(vocab_size = TOTAL_VOCAB,
                              d_model = 1024, d_state = 128,
                              n_priors = 4, n_posteriors = 3),
            SEQ_LEN=128,
            N_STEPS=500,
            SAVE_EVERY=500),
        "512x64": Specification(
            CONFIG=SRLMConfig(vocab_size = TOTAL_VOCAB,
                              d_model = 512, d_state = 64,
                              n_priors=3, n_posteriors=2)),
        "512x64r": Specification(
            CONFIG=SRLMConfig(vocab_size = TOTAL_VOCAB,
                              d_model = 512, d_state = 64,
                              n_priors=3, n_posteriors=2, use_attn_residual=True, use_adaln_in_residual=True)),
        "512h8c128": Specification(
            CONFIG=SRLMConfig(vocab_size = TOTAL_VOCAB,
                              d_model = 512, d_state = 512,
                              n_priors=3, n_posteriors=2,
                              use_attention=True, n_heads=8, context_length=128),
            SEQ_LEN=128),
        "768h12c128": Specification(
            CONFIG=SRLMConfig(vocab_size = TOTAL_VOCAB,
                              d_model = 768, d_state = 512,
                              n_priors=2, n_posteriors=2,
                              use_attention=True, n_heads=12, context_length=128),
            SEQ_LEN=128),
        "512x64rd": Specification(
            CONFIG=SRLMConfig(vocab_size = TOTAL_VOCAB,
                              d_model = 512, d_state = 64,
                              n_priors=4, n_posteriors=4, use_attn_residual=True, use_adaln_in_residual=True)),
        "1024x8r": Specification(
            CONFIG=SRLMConfig(vocab_size = TOTAL_VOCAB,
                              d_model = 1024, d_state = 8,
                              n_priors=3, n_posteriors=2, use_attn_residual=True, use_adaln_in_residual=True)),
        "512x8r": Specification(
            CONFIG=SRLMConfig(vocab_size = TOTAL_VOCAB,
                              d_model = 512, d_state = 8,
                              n_priors=4, n_posteriors=2, use_attn_residual=True, use_adaln_in_residual=True)),
}


def setup(args):
    cwd = Path.cwd()
    print("Current path:", cwd)

    import warnings
    warnings.filterwarnings("error", category=np.ComplexWarning)
    
    #jax.config.update("jax_enable_custom_prng", True)
    #jax.config.update("jax_debug_nans", True)
    #jax.config.update("jax_platform_name", "cpu")
    
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 5)

    # Policy for haiku — cast all compute to bfloat16
    policy = jmp.Policy(
        compute_dtype=jnp.bfloat16,
        param_dtype=jnp.float32,  # keep params in float32
        output_dtype=jnp.bfloat16
    )
    hk.mixed_precision.set_policy(SRLM, policy)

    spec = specifications[args.spec].copy()
    if args.batch:
        spec.BATCH = args.batch
    if args.seq_len:
        spec.SEQ_LEN = args.seq_len
    if args.save_every:
        spec.SAVE_EVERY = args.save_every

    cfg = spec.CONFIG
    if cfg.use_attention:
        assert spec.SEQ_LEN <= cfg.context_length, (
            f"SEQ_LEN={spec.SEQ_LEN} exceeds context_length={cfg.context_length}; "
            f"attention positional embeddings only cover up to context_length positions."
        )

    graph = AbsorbingGraph(VOCAB_SIZE)
    noise = LogLinearNoise()
    sampler = Sampler(graph, noise)
    rng = hk.PRNGSequence(7)

    def model_spec(z, x, sigma, is_training=False):
        assert len(x.shape) == 2, x.shape
        assert len(sigma.shape) == 1, sigma.shape
        hrm = SRLM(spec.CONFIG)
        return hrm(z, x, sigma, is_training)
    model = hk.transform(model_spec)

    print("initializing model...")
    z_init = mk_z(1, 1, spec.CONFIG.d_model)
    x_all = jax.random.randint(next(rng), (1,1), 0, 256)
    sigma = jax.random.normal(next(rng), shape=(1,))
    params = model.init(rng=next(rng), x=x_all, z=z_init, sigma=sigma)
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Parameters: {param_count:,}")
    print(f"Parameter memory: {param_memory_mb(params):.1f} MB")

    z_init = mk_z(spec.BATCH, spec.SEQ_LEN, spec.CONFIG.d_model)
    return Setup(
            cwd,
            spec,
            graph,
            noise,
            sampler,
            rng,
            model,
            params,
            z_init)

@dataclass
class Setup:
    cwd : Any
    spec : Any
    graph : Any
    noise : Any
    sampler : Any
    rng : Any
    model : Any
    params : Any
    z_init : Any

def prepare_for_exam(args, s):
    ckdir = Path(args.checkpoint).absolute()
    ckdir.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()
    params = s.params
    if os.path.exists(ckdir):
        (epoch, step), item = max(list(subitems(ckdir)), key=lambda x: x[0], default=((0,0),None))
        if item is not None:
            params = checkpointer.restore(item, params)
            print(f"epoch {epoch} at step {step}")
    checkpointer.close()
    return params

def prepare_for_train(args, s):
    ckdir = Path(args.checkpoint).absolute()
    ckdir.mkdir(parents=True, exist_ok=True)

    checkpointer = ocp.StandardCheckpointer()
    def restore_checkpoint(params):
        if os.path.exists(ckdir):
            (epoch, step), item = max(list(subitems(ckdir)), key=lambda x: x[0], default=((0,0),None))
            if item is not None:
                params = checkpointer.restore(item, params)
                print(f"Resuming checkpoint: {epoch}.{step}")
            return epoch, step, params
        else:
            checkpointer.save(ckdir / "00000.0", params)
            checkpointer.wait_until_finished()
            return 0, 0, params

    epoch, step, params = restore_checkpoint(s.params)
    def save_checkpoint(params, epoch, step):
        print("Saving epoch", epoch, "step", step)
        path = ckdir / (str(epoch).zfill(5) + "." + str(step))
        if not os.path.exists(path):
            checkpointer.save(path, params)

    print("Setting up optimizer, loss function, training step")
    lr_scheduler = optax.schedules.cosine_decay_schedule(1e-4, 26000 * 3, 0.01) # 500
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.1),
        optax.zero_nans(),
        optax.adamw(lr_scheduler, weight_decay=0.01),
    )
    opt_state = optimizer.init(params)
    loss_fn = loss_function(s.model, s.graph, s.noise)
    @jax.jit
    def train_step_single(key, params, opt_state, x, z, t=None, p_x=None):
        (loss, z), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, key, z, x, t=t, perturbed_batch=p_x)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, z

    @jax.jit
    def train_step_ewc(key, params, opt_state, x, z, params_A, fisher_A, ewc_lambda, t=None, p_x=None):
        def total_loss_fn(p):
            # 1. Base Score Entropy Diffusion Loss
            base_loss, out_z = loss_fn(p, key, z, x, t=t, perturbed_batch=p_x)

            # 2. EWC Penalty
            penalty = ewc_penalty(p, params_A, fisher_A)

            # 3. Combined Loss
            return base_loss + (ewc_lambda / 2.0) * penalty, out_z

        # Take value and gradient of the COMBINED loss
        (loss, z), grads = jax.value_and_grad(total_loss_fn, has_aux=True)(params)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, opt_state, loss, z

    arith_lr_scheduler = optax.schedules.cosine_decay_schedule(1e-5, 26000 * 3, 0.01)
    arith_optimizer = optax.chain(
        optax.clip_by_global_norm(0.1),
        optax.zero_nans(),
        optax.adamw(arith_lr_scheduler, weight_decay=0.01),
    )
    arith_opt_state = arith_optimizer.init(params)

    @jax.jit
    def train_step_arith(key, params, opt_state, x, z, t=None, p_x=None):
        (loss, z), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, key, z, x, t=t, perturbed_batch=p_x)
        updates, opt_state = arith_optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, z

    def train_end():
        checkpointer.close()
        jax.effects_barrier()

    return Trainer(
            ckdir,
            epoch,
            step,
            checkpointer,
            params,
            optimizer,
            opt_state,
            arith_optimizer,
            arith_opt_state,
            loss_fn,
            train_step_single,
            train_step_ewc,
            train_step_arith,
            save_checkpoint,
            train_end)

@dataclass
class Trainer:
    ckdir : Any
    epoch : int
    step : int
    checkpointer : Any
    params : Any
    optimizer : Any
    opt_state : Any
    arith_optimizer : Any
    arith_opt_state : Any
    loss_fn : Any
    train_step_single : Any
    train_step_ewc : Any
    train_step_arith : Any
    save : Any
    end : Any

def subitems(ckdir):
    for subitem in ckdir.iterdir():
        if subitem.name == "progress.json":
            continue
        if subitem.name == "loss.txt":
            continue
        if "." in subitem.name:
            epoch_s, step_s = subitem.name.split(".")
            yield (int(epoch_s), int(step_s)), subitem

def as_text(p):
    return p.astype(np.uint8).tobytes().decode("utf-8", errors="replace")

def from_text(text):
    data = text.encode("utf-8")
    return np.frombuffer(bytearray(data), dtype=np.uint8).astype(np.int16)

def load_wikipedia_finnish(cwd, batch_size, seq_len):
    from wiki_data import WikiDataLoader
    loader = WikiDataLoader(
        parquet_files=[
            cwd / "../../data/train-00000-of-00002.parquet",
            cwd / "../../data/train-00001-of-00002.parquet",
        ],
        batch_size=batch_size,
        seq_len=seq_len,
        seed=42,
    )
    return loader

def load_kalevala(cwd):
    with open(cwd / "../../data/kalevala.plain.txt", "r", encoding="utf-8") as fd:
        text = fd.read().replace("\n", " ")
    raw = np.frombuffer(bytearray(text.encode("utf-8")), dtype=np.uint8).astype(np.int32)
    N   = raw.shape[0]
    print(f"Data: {N} tavua")
    def sample_batch(key, seq_len, batch):
        starts = jax.random.randint(key, (batch,), 0, raw.shape[0] - seq_len)
        starts = np.array(starts)  # bring to CPU
        result = np.stack([raw[s:s+seq_len] for s in starts])
        return jax.device_put(result)
    return sample_batch

parser = argparse.ArgumentParser(
                    prog='srlm',
                    description='Evaluates and trains SRLM -models',
                    epilog='It is a mess.')
parser.set_defaults(run=None)
parser.add_argument("-c", "--checkpoint", help="checkpoint directory")
parser.add_argument("-s", "--spec",
                    default="512x64",
                    help="specification of the model (512x64, 1024x1024)")
parser.add_argument("-l", "--seq_len", type=int)
parser.add_argument("-b", "--batch", type=int)
parser.add_argument("-S", "--save_every", type=int)
subparsers = parser.add_subparsers(help='subcommand help')

def train(args):
    s = setup(args)
    t = prepare_for_train(args, s)
    sample_batch = load_kalevala(s.cwd)
    print(f"Training S5 HRM haiku model | d_model={s.spec.CONFIG.d_model}, d_state={s.spec.CONFIG.d_state}")
    print("-" * 55)
    with open(t.ckdir / "loss.txt", "w") as loss_plot:
        for epoch in range(t.epoch, t.epoch+10):
            total_loss = 0
            for step in range(t.step, s.spec.N_STEPS):
                batch = sample_batch(next(s.rng), s.spec.SEQ_LEN, s.spec.BATCH)
                session_loss = supervision_train(s, t, batch)
                loss_plot.write(f"{step + epoch*s.spec.N_STEPS} {session_loss / s.spec.SUPERVISION}\n")
                loss_plot.flush()
                if t.step % s.spec.STEP_REPORT_EVERY == 0:
                    print(f"  step {step:4d} | loss {session_loss/s.spec.SUPERVISION:.4f}")
                t.step += 1
                total_loss += session_loss
            print(f"epoch {epoch+1}, loss {total_loss / s.spec.N_STEPS / s.spec.SUPERVISION}")
            t.step = 0
            t.save(t.params, epoch+1, 0)
    print("Done.")
    t.end()

parser_train = subparsers.add_parser('train', help='train SRLM from ../../data/kalevala.plaintext.txt')
parser_train.set_defaults(run=train)

def wikitrain(args):
    s = setup(args)
    t = prepare_for_train(args, s)
    loader = load_wikipedia_finnish(s.cwd, s.spec.BATCH, s.spec.SEQ_LEN)
    print(f"Training S5 HRM haiku model | d_model={s.spec.CONFIG.d_model}, d_state={s.spec.CONFIG.d_state}")
    print(f"-" * 55)
    resuming = loader.load_state(t.ckdir / f"progress.json")
    for epoch in range(t.epoch, t.epoch+1):
        with open(t.ckdir / "loss.txt", "a" if resuming else "w") as loss_plot:
            total_loss = 0
            if not resuming:
                loader.shuffle(epoch)
            resuming = False
            while not loader.epoch_done():
                batch = loader.next_batch() # (B, seq_len) int32, or None if epoch done
                if batch is not None:
                    session_loss = supervision_train(s, t, batch)
                    k = loader.steps_this_epoch
                    loss_plot.write(f"{k} {session_loss / s.spec.SUPERVISION}\n")
                    loss_plot.flush()
                    if k % s.spec.STEP_REPORT_EVERY == 0:
                        print(f"{k} | session loss:", session_loss / s.spec.SUPERVISION)
                    if k % s.spec.SAVE_EVERY == 0:
                        print(f"-" * 55)
                        t.save(t.params, t.epoch, loader.steps_this_epoch)
                        loader.save_state(t.ckdir / f"progress.json")
            print(f"last | session loss:", session_loss / s.spec.SUPERVISION)
            print(f"epoch {epoch} done, {loader.steps_this_epoch} steps")
            t.save(t.params, t.epoch+1, 0)
            loader.save_state(t.ckdir / f"progress.json")
    t.end()

def supervision_train(s, t, batch, time=None, p_batch=None):
    z = s.z_init
    session_loss = 0
    for _ in range(s.spec.SUPERVISION):
        t.params, t.opt_state, loss, z = t.train_step_single(next(s.rng), t.params, t.opt_state, batch, z, time, p_batch)
        session_loss += loss
    if np.isnan(session_loss):
        print(f"Training has failed")
        sys.exit(0)
    return session_loss

def arith_supervision_train(s, t, batch, time=None, p_batch=None):
    z = s.z_init
    session_loss = 0
    for _ in range(s.spec.SUPERVISION):
        t.params, t.arith_opt_state, loss, z = t.train_step_arith(next(s.rng), t.params, t.arith_opt_state, batch, z, time, p_batch)
        session_loss += loss
    if np.isnan(session_loss):
        print(f"Training has failed")
        sys.exit(0)
    return session_loss

parser_wikitrain = subparsers.add_parser('wikitrain', help='train from finnish wikipedia')
parser_wikitrain.set_defaults(run=wikitrain)

def wikidry(args):
    i = 0
    s = setup(args)
    t = prepare_for_train(args, s)
    loader = load_wikipedia_finnish(s.cwd, s.spec.BATCH, s.spec.SEQ_LEN)
    loader.shuffle(0)           # shuffle article order for this epoch
    while not loader.epoch_done():
        batch = loader.next_batch() # (B, seq_len) int32, or None if epoch done
        if i % 10000 == 0:
            print(f"at {i}")
        i += 1
    print("total batches: ", i)
    t.end()

parser_wikidry = subparsers.add_parser('wikidry', help='dry run finnish wikipedia')
parser_wikidry.set_defaults(run=wikidry)

def evaluate(args):
    s = setup(args)
    l = s.spec.SEQ_LEN
    params = prepare_for_exam(args, s)
    def projector(x, q):
        q = q[None,:]
        return jnp.where(q == 256, x, q)
    fn = s.sampler.sample2(s.model.apply, projector, 1, l)
    while True:
        query = from_text(input("> "))
        z = mk_z(1, l, s.spec.CONFIG.d_model)
        q = jnp.pad(query, (0, l - len(query)), "constant", constant_values=(256, 256))
        _, x = fn(next(s.rng), params, z, q)
        print(repr(as_text(x[0])))
    t.end()

parser_eval = subparsers.add_parser('eval', help='evaluate on model')
parser_eval.set_defaults(run=evaluate)

def evaluate_m(args):
    s = setup(args)
    l = s.spec.SEQ_LEN
    params = prepare_for_exam(args, s)
    def projector(x, q):
        q = q[None,:]
        return jnp.where(q == 256, x, q)
    fn = s.sampler.sample2(s.model.apply, projector, 1, l)
    z = mk_z(1, l, s.spec.CONFIG.d_model)
    while True:
        query = from_text(input("> "))
        q = jnp.pad(query, (0, l - len(query)), "constant", constant_values=(256, 256))
        z, x = fn(next(s.rng), params, z, q)
        print(repr(as_text(x[0])))
    t.end()

parser_eval_m = subparsers.add_parser('evalm', help='evaluate on model with memory')
parser_eval_m.set_defaults(run=evaluate_m)

def train2(args):
    s = setup(args)
    t = prepare_for_train(args, s)
    sample_batch = load_kalevala(s.cwd)
    print(f"Training S5 HRM haiku model | d_model={s.spec.CONFIG.d_model}, d_state={s.spec.CONFIG.d_state}")
    print("-" * 55)
    with open(t.ckdir / "loss.txt", "w") as loss_plot:
        for epoch in range(t.epoch, t.epoch+2):
            total_loss = 0
            for step in range(t.step, s.spec.N_STEPS // 10):
                batch = sample_batch(next(s.rng), s.spec.SEQ_LEN, s.spec.BATCH)
                base_session_loss = supervision_train(s, t, batch) * 0.7
                base_session_loss = max(10 * s.spec.SUPERVISION, base_session_loss) # goal
                print(f"goal session loss: ({base_session_loss / s.spec.SUPERVISION})")
                session_loss = supervision_train(s, t, batch)
                for i in range(10):
                    if session_loss <= base_session_loss:
                        break
                    if i % 20 == 0:
                        print(f".. showing again ({session_loss / s.spec.SUPERVISION})")
                    session_loss = supervision_train(s, t, batch)
                loss_plot.write(f"{step + epoch*s.spec.N_STEPS // 10} {session_loss / s.spec.SUPERVISION}\n")
                loss_plot.flush()
                if t.step % s.spec.STEP_REPORT_EVERY == 0:
                    print(f"  step {step:4d} | loss {session_loss/s.spec.SUPERVISION:.4f}")
                t.step += 1
                total_loss += session_loss
            print(f"epoch {epoch+1}, loss {total_loss / s.spec.N_STEPS / s.spec.SUPERVISION}")
            t.step = 0
            t.save(t.params, epoch+1, 0)
    print("Done.")
    t.end()

parser_train2 = subparsers.add_parser('train2', help='train SRLM from ../../data/kalevala.plaintext.txt with different method.')
parser_train2.set_defaults(run=train2)

def make_arithmetic_puzzle(key, batch_len, batch_size):
    batch = []
    p_batch = []
    for k in range(batch_size):
        key, key_1 = jax.random.split(key)
        key, key_2 = jax.random.split(key)
        i_1 = jax.random.randint(key_1, (), 0, 100000, dtype=jnp.int32)
        i_2 = jax.random.randint(key_2, (), 0, 100000, dtype=jnp.int32)
        data = from_text(f"{i_1}+{i_2}={str(i_1+i_2)}")
        p_data = from_text(f"{i_1}+{i_2}=")
        data = jnp.pad(data, (0, batch_len - len(data)), constant_values=(0, 0))
        p_data = jnp.pad(p_data, (0, batch_len - len(p_data)), constant_values=(256, 256))
        batch.append(data)
        p_batch.append(p_data)
    return jnp.stack(batch), jnp.stack(p_batch)

def train_arithmetic(args):
    s = setup(args)
    t = prepare_for_train(args, s)
    sample_batch = load_kalevala(s.cwd)
    print(f"Training S5 HRM haiku model | d_model={s.spec.CONFIG.d_model}, d_state={s.spec.CONFIG.d_state}")
    print("-" * 55)
    with open(t.ckdir / "loss.txt", "w") as loss_plot:
        for epoch in range(t.epoch, t.epoch+2):
            total_loss = 0
            for step in range(s.spec.N_STEPS):
                batch, p_batch = make_arithmetic_puzzle(next(s.rng), s.spec.SEQ_LEN, s.spec.BATCH)

                #session_loss = 0
                #for _ in range(s.spec.SUPERVISION):
                #    t.params, t.opt_state, loss, z = t.train_step_ewc(
                #        next(s.rng),
                #        t.params,
                #        t.opt_state,
                #        batch,
                #        z,
                #        params_A,         # <--- Frozen Task A weights
                #        fisher_A,         # <--- Task A parameter importance
                #        ewc_lambda=100.0, # <--- Tune this!
                #        t=time,
                #        p_x=p_batch
                #    )
                #    session_loss += loss
                session_loss = arith_supervision_train(s, t, batch, p_batch=p_batch)
                loss_plot.write(f"{step + epoch*s.spec.N_STEPS} {session_loss / s.spec.SUPERVISION}\n")
                loss_plot.flush()
                if t.step % s.spec.STEP_REPORT_EVERY == 0:
                    print(f"  step {step:4d} | loss {session_loss/s.spec.SUPERVISION:.4f}")
                t.step += 1
                total_loss += session_loss
            print(f"epoch {epoch+1}, loss {total_loss / s.spec.N_STEPS / s.spec.SUPERVISION}")
            t.step = 0
            t.save(t.params, epoch+1, 0)
    print("Done.")
    t.end()

parser_train2 = subparsers.add_parser('train_arithmetic', help='train SRLM from ../../data/kalevala.plaintext.txt with different method.')
parser_train2.set_defaults(run=train_arithmetic)


def compute_empirical_fisher(s, t_trainer, data_loader, num_batches=200):
    print("Computing Fisher Information Matrix...")

    # Initialize Fisher PyTree with zeros (same shape as params)
    fisher = jax.tree_util.tree_map(jnp.zeros_like, t_trainer.params)

    @jax.jit
    def get_squared_grads(params, key, z, batch):
        # We only want the gradient of the loss scalar, so we index [0]
        # to ignore the auxiliary 'z' state returned by loss_fn
        grad_fn = jax.grad(lambda p: t_trainer.loss_fn(p, key, z, batch)[0])
        grads = grad_fn(params)
        return jax.tree_util.tree_map(jnp.square, grads)

    for i in range(num_batches):
        batch = data_loader()
        if batch is None:
            break

        sq_grads = get_squared_grads(t_trainer.params, next(s.rng), s.z_init, batch)

        # Accumulate the moving average
        fisher = jax.tree_util.tree_map(
            lambda f, g: f + (g / num_batches), fisher, sq_grads
        )

        if (i + 1) % 50 == 0:
            print(f"  Fisher computation: {i + 1}/{num_batches} batches done.")

    return fisher

def param_memory_mb(params):
    leaves = jax.tree_util.tree_leaves(params)
    total_bytes = sum(x.size * x.dtype.itemsize for x in leaves)
    return total_bytes / 1024**2

if __name__=="__main__":
    args = parser.parse_args()
    if args.run is None:
        print("no command given")
    else:
        args.run(args)

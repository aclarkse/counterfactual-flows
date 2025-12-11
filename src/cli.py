import typer
from .train_flow import run_train
from .generate_cf import run_generate
from .evaluate import run_eval

app = typer.Typer()

@app.command()
def train(cfg: str):       run_train(cfg)

@app.command()
def cf(cfg: str):          run_generate(cfg)

@app.command()
def eval(cfg: str):        run_eval(cfg)

if __name__ == "__main__":
    app()

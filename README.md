# Assisted Coding Challenges

Tougher programming puzzles from Codeforces, in Kaggle notebooks, with a small AI assistant by your side.
It has read the statement and can explain, give hints and debug, but it won't solve the problem for you.
The companion series without AI is [No-AI Coding Challenges](https://github.com/TortueSagace/no-ai-coding-challenges).

## Using a notebook

- Import the `.ipynb` into Kaggle (or open it in Jupyter) and keep **Internet on**: the first cell clones this repo.
- Get a free key at [openrouter.ai/keys](https://openrouter.ai/keys), then save it in **one** of these two ways:
    - **On Kaggle:** *Add-ons → Secrets* → add `OPENROUTER_API_KEY` → tick it for the notebook.
    - **On your computer:** put `OPENROUTER_API_KEY=...` in a `.env` file next to the notebook.
- In the notebook:
    - `ask("Why is my solution wrong?")`: your solution cell and its sample results are sent with the question.
    - `fix()` as the last line of your solution cell: the cell is replaced by a corrected version.
    - `undo()`, same place: brings your version back. `reset()`: starts a new conversation.
- The assistant only sees the statement, the sample tests and your solution cell, never the other cells.
- Free models may log prompts: don't paste anything private.

## Repository structure

```
assisted-coding-challenges/
├── A1_<problem>_<elo>.ipynb   # challenge notebooks
├── assistant.py               # the AI assistant (model list, system prompt), shared by all notebooks
├── utils.py                   # evaluation utilities, shared by all notebooks
├── acc1_utils.py              # challenge 1: input parsing and answer checking
├── acc1_tests.txt             # challenge 1: hidden tests
├── acc1_answers.txt           # challenge 1: expected answers of the hidden tests
├── acc1_context.md            # challenge 1: what the assistant is told (statement and samples)
└── ...
```

## License

MIT License - Feel free to use and adapt for educational purposes.

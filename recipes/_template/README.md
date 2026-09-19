# <Recipe name>

<One or two sentences: what this recipe produces and why it is worth an
afternoon. No preamble.>

What you will learn: <the two or three things a reader leaves with>. You need
<`ZEROPROOF_API_KEY` / a model endpoint / a Modal account / nothing>; `--dry-run`
needs none of it. <Seconds / minutes / one A10G for ten minutes.>

## Run it

```bash
uv add whileai
cd recipes/<step>/<name>
python run.py                 # the whole recipe
python run.py --dry-run       # offline: no key, no GPU
```

| flag | default | what it does |
|---|---|---|
| `--limit` | all rows | fewer rows, for a smoke run |
| `--dry-run` | off | no model calls and no key |

## What you get

<The output, quoted, with the one number that matters named. A paired number
with a 95% interval on a held-out set — never a mean alone.>

## Next

<The command the reader runs after this one: the next recipe, `wai.push_rows`,
`wai.train`, or the platform page that now has something on it.>

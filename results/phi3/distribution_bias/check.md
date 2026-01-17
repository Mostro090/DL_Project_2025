# Sarcasm Detection Probe — Passive Aggressive Test

## Meta

- Kimi (Groq): `moonshotai/kimi-k2-instruct-0905`
- Llama 70B (Groq): `llama-3.3-70b-versatile`
- Phi-3 (local): `microsoft/Phi-3-mini-4k-instruct`

## Outputs

### The Praise - SHORT (Passive-Aggressive Trap)

**Text**

```text
Very helpful.
```

**Model responses**

| Model | Model ID | Raw output | Parsed (0/1) | Error |
|---|---|---|---:|---|
| kimi | `moonshotai/kimi-k2-instruct-0905` | `0` | 0 |  |
| phi3 | `microsoft/Phi-3-mini-4k-instruct` | `0` | 0 |  |
| llama70b | `llama-3.3-70b-versatile` | `1` | 1 |  |

**Raw (full)**

#### kimi — `moonshotai/kimi-k2-instruct-0905`

```text
0
```

#### phi3 — `microsoft/Phi-3-mini-4k-instruct`

```text
0
```

#### llama70b — `llama-3.3-70b-versatile`

```text
1
```

### The Praise - LONG (Sincere Context)

**Text**

```text
Very helpful. The tutorial explained exactly how to solve the error in just a few steps.
```

**Model responses**

| Model | Model ID | Raw output | Parsed (0/1) | Error |
|---|---|---|---:|---|
| kimi | `moonshotai/kimi-k2-instruct-0905` | `0` | 0 |  |
| phi3 | `microsoft/Phi-3-mini-4k-instruct` | `0` | 0 |  |
| llama70b | `llama-3.3-70b-versatile` | `1` | 1 |  |

**Raw (full)**

#### kimi — `moonshotai/kimi-k2-instruct-0905`

```text
0
```

#### phi3 — `microsoft/Phi-3-mini-4k-instruct`

```text
0
```

#### llama70b — `llama-3.3-70b-versatile`

```text
1
```

### The Reaction - SHORT (Ambiguous)

**Text**

```text
Wow. Just wow.
```

**Model responses**

| Model | Model ID | Raw output | Parsed (0/1) | Error |
|---|---|---|---:|---|
| kimi | `moonshotai/kimi-k2-instruct-0905` | `1` | 1 |  |
| phi3 | `microsoft/Phi-3-mini-4k-instruct` | `0` | 0 |  |
| llama70b | `llama-3.3-70b-versatile` | `1` | 1 |  |

**Raw (full)**

#### kimi — `moonshotai/kimi-k2-instruct-0905`

```text
1
```

#### phi3 — `microsoft/Phi-3-mini-4k-instruct`

```text
0
```

#### llama70b — `llama-3.3-70b-versatile`

```text
1
```

### The Reaction - LONG (Sincere)

**Text**

```text
Wow. Just wow. The view from the top of this mountain is absolutely breathtaking, I have no words.
```

**Model responses**

| Model | Model ID | Raw output | Parsed (0/1) | Error |
|---|---|---|---:|---|
| kimi | `moonshotai/kimi-k2-instruct-0905` | `0` | 0 |  |
| phi3 | `microsoft/Phi-3-mini-4k-instruct` | `0` | 0 |  |
| llama70b | `llama-3.3-70b-versatile` | `0` | 0 |  |

**Raw (full)**

#### kimi — `moonshotai/kimi-k2-instruct-0905`

```text
0
```

#### phi3 — `microsoft/Phi-3-mini-4k-instruct`

```text
0
```

#### llama70b — `llama-3.3-70b-versatile`

```text
0
```

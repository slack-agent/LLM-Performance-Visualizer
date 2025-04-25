import React, { useState, useCallback, useEffect, useMemo } from "react";
import {
  Box,
  Container,
  Paper,
  Typography,
  Slider,
  TextField,
  Grid,
  InputAdornment,
  CssBaseline,
  ThemeProvider,
  createTheme,
} from "@mui/material";
import Plot from "react-plotly.js";



const theme = createTheme({
  palette: {
    primary: { main: "#1a237e" },
    secondary: { main: "#cc8899" },
    background: { default: "#f5f5f5", paper: "#ffffff" },
  },
  typography: { fontFamily: "Inter, Roboto, sans-serif" },
});

// ──────────────────────────────────────────────────────────────────────────────
// Hooks & helpers
// ──────────────────────────────────────────────────────────────────────────────
const useDebounce = (value, delay = 400) => {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(id);
  }, [value, delay]);
  return debounced;
};
const clamp = (v, min, max) => Math.min(Math.max(v, min), max);

// ──────────────────────────────────────────────────────────────────────────────
export default function App() {
  // State
  const [config, setConfig] = useState("");
  const [memoryBandwidth, setMemoryBandwidth] = useState(1792);
  const [computePower, setComputePower] = useState(209);
  const [batchSize, setBatchSize] = useState(1);
  const [maxSeqLen, setMaxSeqLen] = useState(1024);
  const [throughputData, setThroughputData] = useState({ total: { x: [], y: [] }, perUser: { x: [], y: [] } });

  const ranges = useMemo(() => ({
    mem: { min: 0.1, max: 8192, step: 0.1 },
    flops: { min: 0.1, max: 8192, step: 0.1 },
    batch: { min: 1, max: 1024, step: 1 },
    seq: { min: 1024, max: 1048576, step: 1024 },
  }), []);

  // Core calcs (unchanged)
  const calculateModelParameters = useCallback((m) => {
    const { hidden_size, num_hidden_layers, intermediate_size, vocab_size, hidden_act, tie_word_embeddings, num_attention_heads, num_key_value_heads } = m;
    const isGated = ["silu", "swiglu", "geglu"].includes(hidden_act);
    let embed = vocab_size * hidden_size * (tie_word_embeddings ? 1 : 2);
    const head_size = hidden_size / num_attention_heads;
    const attn = 2 * hidden_size ** 2 + 2 * hidden_size * num_key_value_heads * head_size;
    const ffn = (isGated ? 3 : 2) * hidden_size * intermediate_size;
    const ln = 4 * hidden_size;
    return embed + num_hidden_layers * (attn + ffn + ln);
  }, []);

  const calculateOperations = useCallback((m, seq = 1, b = 1) => {
    const {hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, num_key_value_heads, vocab_size, hidden_act} = m;
  
    const isGated = ["silu", "swiglu", "geglu"].includes(hidden_act);
    const actCost = isGated ? 3 : 1;
    const kvScale = num_key_value_heads / num_attention_heads;
  
    // Attention: Q/K/V + output proj + softmax & weighted sum
    const attention =
      2 * hidden_size**2 * (2 + 2 * kvScale) +
      4 * hidden_size * seq;
  
    // FFN: matmuls + activation (+ gate)
    const feed_forward =
      hidden_size * intermediate_size * (isGated ? 6 : 4) +
      intermediate_size * (isGated ? actCost + 1 : actCost);
  
    // LM head
    const lm_head = 2 * hidden_size * vocab_size;
  
    return b * (
      num_hidden_layers * (attention + feed_forward) +
      lm_head
    );
  }, []);
  

  const calculateMemoryReads = useCallback((m, seq = 1, b = 1) => {
    const bytes = { float32: 4, float16: 2, bfloat16: 2 }[m.torch_dtype] ?? 4;
    const kv = b * seq * m.num_hidden_layers * m.hidden_size * 2;
    return (calculateModelParameters(m) + kv) * bytes;
  }, [calculateModelParameters]);

  const calculateThroughput = useCallback((cfg, mem, flops, b, seq) => {
    try {
      const m = JSON.parse(cfg);
      const memT = calculateMemoryReads(m, seq, b) / (mem * 1e9);
      const compT = calculateOperations(m, seq, b) / (flops * 1e12);
      console.log(calculateMemoryReads(m, seq, b)/1e9, calculateOperations(m, seq, b)/1e12);
      return b / (memT + compT);
    } catch {
      return 0;
    }
  }, [calculateOperations, calculateMemoryReads]);

  // Debounced curve update
  const dCfg = useDebounce(config, 200);
  useEffect(() => {
    if (!dCfg) return;
    const mem = memoryBandwidth;
    const fl = computePower;
    const bs = batchSize;
    const ms = maxSeqLen;
    const step = Math.max(1, Math.floor(ms / 800));
    const xs = [], ty = [], py = [];
    for (let s = 1; s <= ms; s += step) {
      const t = calculateThroughput(dCfg, mem, fl, bs, s);
      xs.push(s); ty.push(t); py.push(t / bs);
    }
    setThroughputData({ total: { x: xs, y: ty }, perUser: { x: xs, y: py } });
  }, [dCfg, memoryBandwidth, computePower, batchSize, maxSeqLen, calculateThroughput]);

  // Handlers
  const numChange = (set, r) => (e) => set(clamp(parseFloat(e.target.value) || r.min, r.min, r.max));
  const slideChange = (set) => (_e, v) => set(v);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ minHeight: "100vh", py: 4, bgcolor: "background.default" }}>
        <Container maxWidth="lg">
          {/* Header */}
          <Box textAlign="center" mb={6}>
            <Typography variant="h3" fontWeight={600} color="primary.main" gutterBottom>
              LLM Performance Visualiser
            </Typography>
            <Typography variant="subtitle1" color="text.secondary" maxWidth={800} mx="auto">
              Analyse and visualise transformer throughput across hardware settings.
            </Typography>
          </Box>

          <Grid container spacing={4}>
            {/* Model config */}
            <Grid item xs={12} md={6}>
              <Paper elevation={3} sx={{ p: 3, height: "100%" }}>
                <Typography variant="h6" fontWeight={600} color="primary.main" gutterBottom>
                  Model Configuration
                </Typography>
                <TextField
                  multiline
                  minRows={14}
                  maxRows={14}
                  fullWidth
                  placeholder="Paste model configuration JSON here…"
                  value={config}
                  onChange={(e) => setConfig(e.target.value)}
                  variant="outlined"
                  sx={{ maxHeight: 360 }}
                  InputProps={{ sx: { fontFamily: "monospace", "& textarea": { lineHeight: 1.4 } } }}
                  inputProps={{
                    spellCheck: false,
                    style: { fontFamily: "monospace", lineHeight: 1.4 },
                  }}
                />
              </Paper>
            </Grid>

            {/* Hardware controls */}
            <Grid item xs={12} md={6}>
              <Paper elevation={3} sx={{ p: 3, height: "100%" }}>
                <Typography variant="h6" fontWeight={600} color="primary.main" gutterBottom>
                  Hardware Parameters
                </Typography>
                <ParamControl label="Memory Bandwidth" unit="GB/s" value={memoryBandwidth} range={ranges.mem} integer={false} onNumberChange={numChange(setMemoryBandwidth, ranges.mem)} onSliderChange={slideChange(setMemoryBandwidth)} />
                <ParamControl label="Compute Power" unit="TFLOPS" value={computePower} range={ranges.flops} integer={false} onNumberChange={numChange(setComputePower, ranges.flops)} onSliderChange={slideChange(setComputePower)} />
                <ParamControl label="Batch Size" unit="" value={batchSize} range={ranges.batch} integer onNumberChange={numChange(setBatchSize, ranges.batch)} onSliderChange={slideChange(setBatchSize)} />
                <ParamControl label="Max Sequence Length" unit="" value={maxSeqLen} range={ranges.seq} integer onNumberChange={numChange(setMaxSeqLen, ranges.seq)} onSliderChange={slideChange(setMaxSeqLen)} />
              </Paper>
            </Grid>

            {/* Chart */}
            <Grid item xs={12}>
              <Paper elevation={3} sx={{ p: 3 }}>
                <Typography variant="h6" fontWeight={600} color="primary.main" gutterBottom>
                  Throughput vs Sequence Length
                </Typography>
                <Plot
                  data={[
                    { type: "scatter", mode: "lines", x: throughputData.perUser.x, y: throughputData.perUser.y, name: "Per‑User", line: { width: 2, color: theme.palette.primary.main } },
                    { type: "scatter", mode: "lines", x: throughputData.total.x, y: throughputData.total.y, name: "Total", line: { width: 2, color: theme.palette.secondary.main } },
                  ]}
                  layout={{ autosize: true, margin: { t: 40, r: 30, b: 50, l: 60 }, paper_bgcolor: theme.palette.background.paper, plot_bgcolor: theme.palette.background.paper, xaxis: { title: "Sequence Length", gridcolor: "#e0e0e0" }, yaxis: { title: "Throughput (tokens/s)", gridcolor: "#e0e0e0" }, legend: { orientation: "h", x: 0, y: 1.1 }, hovermode: "x unified" }}
                  config={{ displaylogo: false, responsive: true }}
                  style={{ width: "100%", height: "600px" }}
                />
              </Paper>
            </Grid>
          </Grid>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

// ──────────────────────────────────────────────────────────────────────────────
function ParamControl({ label, unit, value, range: { min, max, step }, onNumberChange, onSliderChange, integer }) {
  return (
    <Box mb={3}>
      <Typography fontWeight={500} color="text.primary" gutterBottom>
        {label}: {value.toLocaleString()} {unit}
      </Typography>
      <Grid container spacing={2} alignItems="center">
        <Grid item xs>
          <Slider value={value} min={min} max={max} step={step} valueLabelDisplay="auto" onChange={onSliderChange} />
        </Grid>
        <Grid item>
          <TextField
            type="number"
            size="small"
            value={value}
            onChange={onNumberChange}
            inputProps={{ min, max, step, inputMode: "decimal", style: { textAlign: "center" } }}
            InputProps={{
              endAdornment: unit && <InputAdornment position="end">{unit}</InputAdornment>,
              sx: { width: 160 },
            }}
          />
        </Grid>
      </Grid>
    </Box>
  );
}

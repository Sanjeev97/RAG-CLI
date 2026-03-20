const logEl = document.getElementById('log');
const statusEl = document.getElementById('status');
const modelPathEl = document.getElementById('modelPath');
const dataDirEl = document.getElementById('dataDir');
const showSourcesEl = document.getElementById('showSources');
const queryEl = document.getElementById('query');
const btnSendEl = document.getElementById('btnSend');
const btnInitEl = document.getElementById('btnInit');
const btnStopEl = document.getElementById('btnStop');

let initJobId = null;
let chatJobId = null;
let pipelineReady = false;

function setStatus(text){ statusEl.textContent = text; }

function appendMsg(text, cls="system"){
  const div = document.createElement('div');
  div.className = `msg ${cls}`;
  div.textContent = text;
  logEl.appendChild(div);
  logEl.scrollTop = logEl.scrollHeight;
  return div;
}

function appendSources(sources){
  const box = document.createElement('div');
  box.className = 'sources';
  if(!sources || sources.length === 0){
    box.textContent = '(no sources)';
    return box;
  }
  for(const s of sources){
    const rel = (typeof s.similarity_score === 'number') ? s.similarity_score.toFixed(2) : String(s.similarity_score);
    const el = document.createElement('div');
    el.textContent = `- ${s.source} (relevance: ${rel})`;
    box.appendChild(el);
  }
  return box;
}

function confidenceBar(conf){
  if(typeof conf !== 'number') return '';
  const filled = Math.max(0, Math.min(10, Math.floor(conf * 10)));
  const empty = 10 - filled;
  const bar = '█'.repeat(filled) + '░'.repeat(empty);
  return `${bar} ${(conf*100).toFixed(0)}%`;
}

async function waitInit(){
  if(!initJobId) return;
  while(true){
    const r = await fetch(`/api/init_status?id=${encodeURIComponent(initJobId)}`);
    const data = await r.json();
    if(data.status === 'running'){
      setStatus(data.message || 'Initializing...');
      await new Promise(res => setTimeout(res, 1200));
      continue;
    }
    if(data.status === 'done'){
      pipelineReady = true;
      btnSendEl.disabled = false;
      btnInitEl.disabled = true;
      btnStopEl.disabled = false;
      setStatus('Ready');
      appendMsg('System ready.', 'system');
      if(typeof data.show_sources === 'boolean'){
        showSourcesEl.checked = data.show_sources;
      }
      if(data.help_text){
        appendMsg(data.help_text, 'system');
      }
      return;
    }
    if(data.status === 'failed'){
      setStatus('Initialization failed');
      btnSendEl.disabled = true;
      btnInitEl.disabled = false;
      btnStopEl.disabled = true;
      appendMsg('Initialization error: ' + (data.error || 'Unknown error'), 'system');
      return;
    }
  }
}

async function pollChat(){
  if(!chatJobId) return;
  while(true){
    const r = await fetch(`/api/chat_status?id=${encodeURIComponent(chatJobId)}`);
    const data = await r.json();
    if(data.status === 'running'){
      setStatus(data.message || 'Thinking...');
      await new Promise(res => setTimeout(res, 600));
      continue;
    }
    if(data.status === 'done'){
      setStatus('Ready');
      const msg = appendMsg('Assistant: ' + (data.answer || ''), 'assistant');
      if(data.sources && showSourcesEl.checked){
        msg.appendChild(appendSources(data.sources));
      }
      if(typeof data.confidence === 'number' && data.confidence > 0){
        const confDiv = document.createElement('div');
        confDiv.className = 'confidence';
        confDiv.textContent = 'Confidence: ' + confidenceBar(data.confidence);
        msg.appendChild(confDiv);
      }
      if(typeof data.elapsed === 'number'){
        appendMsg('Response time: ' + data.elapsed.toFixed(1) + 's', 'system');
      }
      if(typeof data.show_sources === 'boolean'){
        showSourcesEl.checked = data.show_sources;
      }
      btnSendEl.disabled = false;
      chatJobId = null;
      return;
    }
    if(data.status === 'failed'){
      setStatus('Error');
      appendMsg('Chat error: ' + (data.error || 'Unknown error'), 'system');
      btnSendEl.disabled = false;
      chatJobId = null;
      return;
    }
  }
}

async function startInit(){
  const modelPath = modelPathEl.value.trim();
  const dataDir = dataDirEl.value.trim();
  if(!modelPath){
    setStatus('Initialization failed');
    appendMsg('Initialization error: model path not configured.', 'system');
    return;
  }
  if(!dataDir){
    setStatus('Initialization failed');
    appendMsg('Initialization error: data folder not configured.', 'system');
    return;
  }

  btnInitEl.disabled = true;
  btnSendEl.disabled = true;
  btnStopEl.disabled = false;
  setStatus('Initializing...');
  appendMsg('Initializing RAG system...', 'system');

  const payload = {
    model_path: modelPath,
    data_dir: dataDir,
    show_sources: !!showSourcesEl.checked
  };
  const r = await fetch('/api/init', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) });
  const data = await r.json();
  if(!r.ok){
    setStatus('Initialization failed');
    btnInitEl.disabled = false;
    btnStopEl.disabled = true;
    appendMsg('Initialization error: ' + (data.error || 'Unknown error'), 'system');
    return;
  }
  initJobId = data.id;
  await waitInit();
}

async function startChat(){
  const query = queryEl.value.trim();
  if(!query) return;
  if(!pipelineReady){ alert('Initialize the system first'); return; }
  if(chatJobId) return;

  btnSendEl.disabled = true;
  setStatus('Thinking...');
  appendMsg('You: ' + query, 'user');
  queryEl.value = '';

  const r = await fetch('/api/chat', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ query }) });
  const data = await r.json();
  if(!r.ok){
    btnSendEl.disabled = false;
    setStatus('Error');
    appendMsg('Chat error: ' + (data.error || 'Unknown error'), 'system');
    return;
  }
  chatJobId = data.id;
  await pollChat();
}

btnInitEl.addEventListener('click', startInit);
btnSendEl.addEventListener('click', startChat);
queryEl.addEventListener('keydown', (e) => {
  if(e.key === 'Enter' && !e.shiftKey){
    e.preventDefault();
    startChat();
  }
});

btnStopEl.addEventListener('click', () => {
  // Lightweight stop: clear UI. Background job may still complete server-side.
  logEl.textContent = '';
  appendMsg('Stopped (current job may still be running).', 'system');
  btnSendEl.disabled = !pipelineReady;
  setStatus('Idle');
});

appendMsg('Local RAG Web UI loaded.', 'system');
appendMsg('Auto-initializing RAG system...', 'system');
appendMsg('Tip: commands like /help are supported.', 'system');
startInit();


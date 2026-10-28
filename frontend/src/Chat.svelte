<script>
  let textInput = '';
  let tokens = [];
  let verdict = null;
  let isAnalyzing = false;
  let error = null;
  let analysisModel = 'phase2';
  // agent_mode: 'model' | 'agent' | 'both'
  let agentMode = 'both';

  let statuses = [];
  let llmContent = '';
  let reasoningContent = '';

  // reference to auto-scroll container
  let scrollContainer;

  // Word count
  $: wordCount = textInput.trim() ? textInput.trim().split(/\s+/).length : 0;
  $: isTooShort = wordCount > 0 && wordCount < 30;

  async function handleAnalyze() {
    if (!textInput.trim()) return;
    
    const payloadText = textInput;
    textInput = ''; // Clear immediately
    
    isAnalyzing = true;
    tokens = [];
    verdict = null;
    error = null;
    statuses = [];
    llmContent = '';
    reasoningContent = '';

    try {
      const response = await fetch('/analyze_stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: payloadText, model: analysisModel, agent_mode: agentMode })
      });

      if (!response.ok) {
        throw new Error('Analysis failed.');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      
      let buffer = '';
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        
        let parts = buffer.split('\n\n');
        buffer = parts.pop();
        
        for (const part of parts) {
          if (!part.trim()) continue;
          try {
            const data = JSON.parse(part);
            if (data.type === 'verdict') {
              verdict = data;
            } else if (data.type === 'token') {
              tokens = [...tokens, data.token];
            } else if (data.type === 'status') {
              statuses = [...statuses, data.message];
            } else if (data.type === 'llm_chunk') {
              if (data.reasoning) reasoningContent += data.reasoning;
              if (data.content) llmContent += data.content;
            } else if (data.type === 'error') {
              error = data.message;
            }
            
            // Auto scroll explicitly on layout updates
            setTimeout(() => {
              if (scrollContainer) scrollContainer.scrollTop = scrollContainer.scrollHeight;
            }, 10);
          } catch (e) {
            console.error('Error parsing SSE part:', e);
          }
        }
      }
      
    } catch (err) {
      error = err.message || 'An error occurred';
    } finally {
      isAnalyzing = false;
    }
  }

  function getTierColor(tier) {
    switch(tier) {
      case 4: return 'var(--highlight-critical)';
      case 3: return 'var(--highlight-high)';
      case 2: return 'var(--highlight-moderate)';
      case 1: return 'var(--highlight-low)';
      default: return 'transparent';
    }
  }

  function getTierTextColor(tier) {
    if (tier === 4) return 'var(--highlight-text-critical)';
    return 'var(--text-main)';
  }

  function handleKeydown(event) {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
      handleAnalyze();
    }
  }
</script>

<div class="chat-container">
  <div class="chat-bg"></div>

  <div class="stream-area" bind:this={scrollContainer}>
    <div class="reading-width">
      
      {#if tokens.length === 0 && !isAnalyzing && !verdict}
        <div class="empty-state animate-fade-in">
          <div class="eyebrow"><span class="eyebrow-line"></span> ARTICLE CHECK</div>
          <h1 class="super-title">Check the story.<br /><span>Read the signals.</span></h1>
          <p class="subtitle">Paste an article to review its probabilities, highlighted language, and optional agent notes.</p>
        </div>
      {/if}

      {#if statuses.length > 0}
        <div class="loader-container animate-fade-in">
          {#each statuses as status, i}
             <div class="status-item animate-fade-in" style="opacity: {(i === statuses.length - 1 && isAnalyzing) ? 1 : 0.7}">
               {#if i === statuses.length - 1 && isAnalyzing}
                 <span class="spinner"></span>
               {:else}
                 <span class="check">✓</span>
               {/if} 
               {status}
             </div>
          {/each}
        </div>
      {/if}

      {#if verdict}
        <div class="verdict-card animate-fade-in">
          <div class="verdict-banner">
            <h1 class="{verdict.label === 'FAKE' ? 'text-critical' : 'text-safe'}">{verdict.label}</h1>
          </div>
          <div class="verdict-details">
            <div class="prob-row">
              <span class="prob-label">Real</span>
              <div class="bar-bg"><div class="bar bg-safe" style="width: {verdict.prob_real * 100}%"></div></div>
              <span class="prob-val">{(verdict.prob_real * 100).toFixed(1)}%</span>
            </div>
            <div class="prob-row">
              <span class="prob-label">Fake</span>
              <div class="bar-bg"><div class="bar bg-critical" style="width: {verdict.prob_fake * 100}%"></div></div>
              <span class="prob-val">{(verdict.prob_fake * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
      {/if}

      {#if tokens.length > 0}
        <div class="legend animate-fade-in">
          <span class="legend-title">Attention Weight</span>
          <div class="legend-swatches">
            <span><div class="swatch" style="background: var(--highlight-critical);"></div> Critical</span>
            <span><div class="swatch" style="background: var(--highlight-high);"></div> High</span>
            <span><div class="swatch" style="background: var(--highlight-moderate);"></div> Mod</span>
            <span><div class="swatch" style="background: var(--highlight-low);"></div> Low</span>
          </div>
        </div>

        <div class="token-stream animate-fade-in">
          {#each tokens as token, i}
             <span 
                class="token tier-{token.tier}" 
                style="background-color: {getTierColor(token.tier)}; color: {getTierTextColor(token.tier)};"
             >{token.leading_space ? ' ' : ''}{token.word}</span>
          {/each}
        </div>
      {/if}

      {#if reasoningContent || llmContent}
        <div class="llm-container animate-fade-in">
           {#if reasoningContent}
             <details class="reasoning-block">
               <summary>Intelligent Agent Thought Process</summary>
               <div class="reasoning-text">{reasoningContent}</div>
             </details>
           {/if}
           {#if llmContent}
             <div class="llm-output">{llmContent}</div>
           {/if}
        </div>
      {/if}

      {#if error}
        <div class="error-banner animate-fade-in">
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
          {error}
        </div>
      {/if}
      
      <!-- Spacing block for input -->
      <div style="height: 360px;"></div>
    </div>
  </div>

  <div class="input-area">
    <div class="input-card card">
      <div class="input-heading"><strong>Article text</strong><span>Ctrl + Enter to analyze</span></div>
      <textarea 
        bind:value={textInput} 
        on:keydown={handleKeydown}
        placeholder="Paste an article or headline…"
        disabled={isAnalyzing}
      ></textarea>

      {#if isTooShort}
        <div class="short-text-warning">
          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
          <span><strong>Text too short ({wordCount} words).</strong> This model was trained on full news articles (200+ words). Short factual statements are <em>out-of-distribution</em> and will produce unreliable results. Paste a complete news article or paragraph for accurate detection.</span>
        </div>
      {:else if wordCount > 0}
        <div class="word-count-ok">{wordCount} words</div>
      {/if}
      
      <div class="input-footer">
        <div class="footer-left">
          <button class="icon-btn" on:click={() => {textInput = ''; tokens = []; verdict = null;}} title="Clear" disabled={isAnalyzing}>
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"></path><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
          </button>
          
          <div class="model-toggle-inline">
            <span class="label">Model:</span>
            <select bind:value={analysisModel} class="select-clean" disabled={agentMode === 'agent'}>
              <option value="phase1">Phase 1 Data</option>
              <option value="phase2">Phase 2 Fusion</option>
            </select>
          </div>

          <div class="agent-toggle-inline">
            <span class="label">Agent:</span>
            <div class="agent-pill-group" role="group" aria-label="AI Agent mode">
              <button
                class="agent-pill {agentMode === 'model' ? 'active' : ''}"
                on:click={() => agentMode = 'model'}
                title="Use trained model only — no LLM reasoning"
                disabled={isAnalyzing}
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a4 4 0 0 1 4 4v1H8V6a4 4 0 0 1 4-4z"/><rect x="2" y="7" width="20" height="14" rx="3"/><circle cx="8" cy="14" r="1.5"/><circle cx="16" cy="14" r="1.5"/></svg>
                Model
              </button>
              <button
                class="agent-pill {agentMode === 'agent' ? 'active' : ''}"
                on:click={() => agentMode = 'agent'}
                title="Use AI agent reasoning only — skips model scoring"
                disabled={isAnalyzing}
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
                Agent
              </button>
              <button
                class="agent-pill {agentMode === 'both' ? 'active' : ''}"
                on:click={() => agentMode = 'both'}
                title="Use both model scoring AND AI agent reasoning"
                disabled={isAnalyzing}
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                Both
              </button>
            </div>
          </div>
        </div>
        
        <button class="btn btn-primary submit-btn" on:click={handleAnalyze} disabled={isAnalyzing || !textInput.trim()}>
          {#if isAnalyzing}
            <div class="spinner"></div>
          {:else}
            Analyze
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-left: 6px;"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
          {/if}
        </button>
      </div>
    </div>
  </div>

</div>

<style>
  .chat-container {
    display: flex;
    flex-direction: column;
    height: 100vh;
    position: relative;
    background-color: transparent; /* allow global body or layer colors to breathe */
    overflow: hidden;
  }

  .chat-bg {
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 15% 30%, rgba(59, 130, 246, 0.06) 0%, var(--bg-color) 45%, rgba(239, 68, 68, 0.04) 100%);
    opacity: 1;
    z-index: 0;
    pointer-events: none;
  }

  .stream-area {
    flex: 1;
    overflow-y: auto;
    padding: 3rem 1.5rem 0 1.5rem;
    position: relative;
    z-index: 10;
  }

  .reading-width {
    max-width: 768px;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    justify-content: center;
    text-align: left;
    margin-top: 20vh;
  }

  .empty-state .super-title {
    font-size: 5.5rem;
    line-height: 0.9;
    margin-bottom: 2rem;
    letter-spacing: -2px;
  }

  .outline-text {
    color: transparent;
    -webkit-text-stroke: 1px var(--border-strong);
  }

  .empty-state .subtitle {
    font-size: 1.1rem;
    color: var(--text-muted);
    max-width: 500px;
    line-height: 1.8;
  }

  .bg-critical { background-color: var(--accent); }
  .bg-safe { background-color: var(--success); }
  .text-critical { color: var(--accent); }
  .text-safe { color: var(--success); }

  .verdict-card {
    background: transparent;
    margin-bottom: 2rem;
  }

  .verdict-banner {
    padding: 1rem 0;
    text-align: left;
  }

  .verdict-banner h1 {
    font-size: 1.8rem;
    letter-spacing: 2px;
  }

  .verdict-details {
    padding: 0;
  }

  .prob-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.8rem;
  }

  .prob-label {
    width: 40px;
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--text-muted);
  }

  .prob-val {
    width: 50px;
    text-align: right;
    font-weight: 700;
    font-size: 0.9rem;
  }

  .bar-bg {
    flex: 1;
    height: 8px;
    background: var(--border-strong);
    border-radius: 4px;
    overflow: hidden;
  }

  .bar {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
  }

  .legend {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
    padding: 0;
    background: transparent;
    border: none;
    box-shadow: none;
  }

  .legend-title {
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .legend-swatches {
    display: flex;
    gap: 1.2rem;
  }

  .legend-swatches span {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--text-main);
  }

  .swatch {
    width: 12px;
    height: 12px;
    border-radius: 3px;
  }

  .token-stream {
    background: transparent;
    padding: 0;
    border: none;
    box-shadow: none;
    line-height: 2;
    font-size: 1.05rem;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .token {
    border-radius: 3px;
    transition: background-color 0.2s;
    padding: 2px 0;
  }

  .error-banner {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 1rem;
    background: #fef2f2;
    border: 1px solid #fecaca;
    color: #991b1b;
    border-radius: var(--radius-md);
    font-weight: 500;
    margin-top: 1rem;
  }

  .input-area {
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    padding: 0 1.5rem 1.5rem 1.5rem;
    display: flex;
    justify-content: center;
    background: linear-gradient(to top, var(--bg-color) 60%, transparent);
    pointer-events: none;
    z-index: 50;
  }

  .input-card {
    width: 100%;
    max-width: 768px;
    display: flex;
    flex-direction: column;
    padding: 0.8rem;
    border-radius: var(--radius-lg);
    box-shadow: none;
    pointer-events: auto;
    border: 1px solid var(--border-strong); 
    background: var(--bg-color);
  }

  textarea {
    width: 100%;
    min-height: 50px;
    max-height: 200px;
    border: none;
    background: transparent;
    padding: 0.5rem;
    font-size: 1rem;
    font-family: inherit;
    color: var(--text-main);
    resize: vertical;
    outline: none;
  }

  textarea::placeholder {
    color: #9ca3af;
  }

  .short-text-warning {
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
    background: rgba(234, 179, 8, 0.08);
    border: 1px solid rgba(234, 179, 8, 0.3);
    border-radius: var(--radius-md);
    padding: 0.55rem 0.8rem;
    margin: 0.3rem 0.4rem;
    font-size: 0.78rem;
    color: #fbbf24;
    line-height: 1.5;
  }

  .short-text-warning svg {
    flex-shrink: 0;
    margin-top: 2px;
    color: #fbbf24;
  }

  .short-text-warning strong {
    color: #fde68a;
  }

  .short-text-warning em {
    font-style: italic;
    color: #f59e0b;
  }

  .word-count-ok {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-align: right;
    padding: 0 0.6rem 0.1rem 0;
    opacity: 0.7;
  }


  .input-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 0.5rem;
    padding-top: 0.5rem;
    border-top: 1px solid var(--border-strong);
  }

  .footer-left {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .model-toggle-inline {
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }

  .model-toggle-inline .label,
  .agent-toggle-inline .label {
    font-size: 0.75rem;
    color: var(--text-muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    white-space: nowrap;
  }

  .select-clean {
    border: none;
    background: transparent;
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-main);
    outline: none;
    cursor: pointer;
    font-family: inherit;
  }

  /* Style options so they don't blend with background text when dropdown is open */
  .select-clean option {
    background: var(--bg-color);
    color: var(--text-main);
    padding: 0.5rem;
  }

  .select-clean:disabled {
    opacity: 0.35;
    cursor: not-allowed;
  }

  /* Agent mode pill group */
  .agent-toggle-inline {
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }

  .agent-pill-group {
    display: flex;
    align-items: center;
    background: var(--bg-color);
    border: 1px solid var(--border-strong);
    border-radius: 999px;
    padding: 2px;
    gap: 2px;
  }

  .agent-pill {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 3px 10px;
    border-radius: 999px;
    border: none;
    background: transparent;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-muted);
    cursor: pointer;
    font-family: inherit;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    transition: background 0.2s, color 0.2s, box-shadow 0.2s;
    white-space: nowrap;
  }

  .agent-pill:hover:not(:disabled):not(.active) {
    background: var(--border);
    color: var(--text-main);
  }

  .agent-pill.active {
    background: var(--primary);
    color: #fff;
    box-shadow: 0 0 8px rgba(99, 102, 241, 0.45);
  }

  .agent-pill:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  .icon-btn {
    background: transparent;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    padding: 0.4rem;
    border-radius: var(--radius-button);
    display: flex;
    align-items: center;
    transition: all 0.2s;
  }

  .icon-btn:hover {
    background: var(--border);
    color: var(--text-main);
  }

  .submit-btn {
    padding: 0.5rem 1rem;
    white-space: nowrap;
  }

  .spinner {
    width: 16px;
    height: 16px;
    border: 2px solid rgba(255,255,255,0.3);
    border-top-color: #ffffff;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  /* Multi-Step Loader */
  .loader-container {
     margin-top: 2rem;
     display: flex;
     flex-direction: column;
     gap: 1.5rem;
  }
  .status-item {
     display: flex;
     align-items: center;
     gap: 1rem;
     font-size: 1.1rem;
     font-family: 'CookConthic', sans-serif;
     letter-spacing: 1px;
     text-transform: uppercase;
  }
  .check {
     color: var(--success);
     font-weight: bold;
     font-size: 1.2rem;
  }

  /* LLM Content */
  .llm-container {
    margin-top: 3rem;
    padding: 2rem;
    background: transparent;
    border: 1px dashed var(--border-strong);
    border-radius: var(--radius-md);
  }
  .reasoning-block {
    margin-bottom: 2rem;
  }
  .reasoning-block summary {
    cursor: pointer;
    color: var(--primary);
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
    outline: none;
  }
  .reasoning-text {
    margin-top: 1rem;
    padding: 1rem;
    border-left: 2px solid var(--border-strong);
    color: var(--text-muted);
    font-size: 0.95rem;
    white-space: pre-wrap;
    line-height: 1.6;
  }
  .llm-output {
    font-size: 1.1rem;
    color: var(--text-main);
    line-height: 1.8;
    white-space: pre-wrap;
  }

  /* Responsive fixes for Chat */
  @media (max-width: 768px) {
    .empty-state .super-title {
      font-size: 11vw;
    }
    
    .input-area {
      padding: 0 1rem 1rem 1rem;
    }
    
    .input-footer {
      flex-direction: column;
      gap: 1.2rem;
      align-items: stretch;
    }

    .footer-left {
      flex-direction: column;
      align-items: flex-start;
      gap: 1.2rem;
      width: 100%;
    }

    .model-toggle-inline,
    .agent-toggle-inline {
      width: 100%;
      justify-content: flex-start;
      margin-bottom: 0.5rem;
    }

    .agent-pill-group {
      width: 100%;
      flex-wrap: nowrap;
      justify-content: space-between;
    }

    .agent-pill {
      flex: 1;
      justify-content: center;
    }

    .submit-btn {
      width: 100%;
      justify-content: center;
      padding: 0.8rem;
    }

    .verdict-banner h1 {
      font-size: 1.5rem;
    }

    .prob-row {
      flex-direction: column;
      align-items: flex-start;
      gap: 0.2rem;
    }
    
    .prob-label {
      width: 100%;
    }
    
    .prob-val {
      width: 100%;
      text-align: left;
    }
  }
  /* Article review layout */
  .chat-container { font-family:Inter,"Segoe UI",Arial,sans-serif; }
  .chat-bg { background:radial-gradient(ellipse at 75% 0%,rgba(59,130,246,.07),transparent 36%),#090b10; }
  .stream-area { padding: 5rem 1.5rem 0; }
  .reading-width { max-width: 1060px; }
  .empty-state { margin-top: 4vh; margin-bottom: 2rem; padding-bottom: 1.5rem; }
  .eyebrow { display:flex; align-items:center; gap:.65rem; color:#8c9ab1; font-family:inherit; font-size:.66rem; font-weight:650; letter-spacing:.13em; }
  .eyebrow-line { width:24px; height:1px; background:#7aa9ff; }
  .empty-state .super-title { font-family:inherit; font-size:clamp(2.4rem,4.8vw,4rem); line-height:1.06; margin:1.15rem 0 .8rem; letter-spacing:-.055em; font-weight:650; }
  .empty-state .super-title span { color:#84aefb; }
  .empty-state .subtitle { font-family:inherit; font-size:.94rem; max-width:580px; line-height:1.65; }
  .input-area { padding:0 1.5rem 1.25rem; background:linear-gradient(to top,#090b10 78%,rgba(9,11,16,.82) 91%,transparent); }
  .input-card { max-width:1060px; padding:1.1rem 1.2rem .9rem; border-radius:16px; box-shadow:0 18px 65px rgba(0,0,0,.35); border:1px solid rgba(160,180,220,.2); background:rgba(13,16,23,.97); }
  .input-heading { display:flex; justify-content:space-between; align-items:center; padding:0 .35rem .65rem; color:#8792a5; font-size:.72rem; }
  .input-heading strong { font-size:.88rem; font-weight:600; color:#e9edf5; }
  textarea { min-height:220px; max-height:min(38vh,420px); padding:1rem; border:1px solid rgba(160,180,220,.12); border-radius:10px; background:rgba(255,255,255,.018); font-family:inherit; font-size:.94rem; line-height:1.65; overflow-y:auto; }
  textarea:focus { border-color:rgba(122,169,255,.48); box-shadow:0 0 0 3px rgba(122,169,255,.08); }
  .word-count-ok { padding:.35rem .25rem .1rem 0; }
  @media(max-width:768px) {
    .stream-area { padding:5rem 1rem 0; }
    .empty-state .super-title { font-size:10vw; }
    .input-area { padding:0 .75rem .75rem; }
    textarea { min-height:190px; max-height:34vh; }
    .input-heading { padding-left:.1rem; padding-right:.1rem; }
  }

  @media(max-height:920px) {
    .stream-area { padding-top:4.25rem; }
    .empty-state { margin-top:0; margin-bottom:.75rem; }
    .empty-state .super-title { font-size:clamp(2.2rem,4.3vw,3.4rem); margin:.8rem 0 .55rem; }
    textarea { min-height:180px; max-height:30vh; }
  }

  @media(max-width:900px) {
    .input-footer { flex-wrap:wrap; gap:.75rem; }
    .footer-left { width:auto; flex-direction:row; flex-wrap:wrap; align-items:center; gap:.7rem; }
    .submit-btn { width:auto; }
  }

  @media(max-width:520px) {
    .empty-state .super-title { font-size:2.35rem; }
    .input-heading { font-size:.64rem; }
    .input-footer { align-items:stretch; }
    .footer-left { width:100%; flex-direction:column; align-items:flex-start; }
    .model-toggle-inline,.agent-toggle-inline { width:100%; }
    .submit-btn { width:100%; }
  }
</style>

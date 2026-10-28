<script>
  export let navigate;

  const metrics = [
    { value: '99.42%', label: 'Test accuracy', detail: 'Phase 2 fusion' },
    { value: '99.42%', label: 'Macro F1', detail: 'Phase 2 fusion' }
  ];

  function openDetector() {
    navigate('/chat');
  }
</script>

<main class="overview-page">
  <div class="overview-glow" aria-hidden="true"></div>

  <section class="welcome-section">
    <div class="welcome-copy">
      <p class="eyebrow">TRUTH SHIELD <span>/</span> NEWS CHECK</p>
      <h1>Check a news<br /><span>article.</span></h1>
      <p class="welcome-description">See the model score, highlighted language, and optional agent notes for an article.</p>
      <button class="primary-action" on:click={openDetector}>
        Analyze an article
        <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 10h11M10 4l6 6-6 6" /></svg>
      </button>
    </div>

    <button class="workflow-card" on:click={openDetector} aria-label="Open article analysis">
      <div class="workflow-header"><span>HOW IT WORKS</span><span>01—03</span></div>
      <div class="workflow-step"><span>01</span><div><strong>Paste an article</strong><small>Use the full text when possible.</small></div></div>
      <div class="workflow-step"><span>02</span><div><strong>Choose how to check</strong><small>Model, agent, or both.</small></div></div>
      <div class="workflow-step"><span>03</span><div><strong>Review the result</strong><small>Verdict, probabilities, and text signals.</small></div></div>
      <div class="workflow-link">Open detector <span>↗</span></div>
    </button>
  </section>

  <section class="metrics-section" aria-label="Recorded model evaluation results">
    <div class="section-heading">
      <div><p class="eyebrow">MODEL RESULTS</p><h2>Recorded test scores</h2></div>
      <p class="source-note">Phase 2 · previously recorded evaluation</p>
    </div>
    <div class="metrics-grid">
      {#each metrics as metric}
        <article class="metric-card"><span class="metric-label">{metric.label}</span><strong>{metric.value}</strong><span class="metric-detail">{metric.detail}</span></article>
      {/each}
    </div>
  </section>

  <footer class="overview-footer"><span>TRUTH SHIELD</span><span>Check sources before sharing.</span></footer>
</main>

<style>
  .overview-page{--ink:#f4f5f7;--muted:#a4aab5;--line:rgba(182,191,215,.14);--blue:#8db4ff;position:relative;isolation:isolate;min-height:100%;overflow:hidden;padding:94px clamp(22px,6.5vw,112px) 28px;color:var(--ink);background:#090b10;font-family:Inter,"Segoe UI",Arial,sans-serif}
  .overview-glow{position:absolute;z-index:-1;pointer-events:none;inset:0;background:radial-gradient(ellipse at 73% 7%,rgba(68,91,169,.12),transparent 34%)}
  .welcome-section{max-width:1180px;margin:0 auto;display:grid;grid-template-columns:1.1fr .9fr;gap:clamp(36px,7vw,100px);align-items:center;padding:42px 0 60px}
  .eyebrow{font-family:inherit;font-size:.68rem;letter-spacing:.12em;font-weight:650;color:#929bad;display:flex;align-items:center;gap:10px}
  .eyebrow span{color:#59667c}
  .welcome-copy h1{font-family:inherit;font-size:clamp(2.8rem,5.6vw,5rem);letter-spacing:-.055em;line-height:1.05;font-weight:650;margin:23px 0 18px}
  .welcome-copy h1 span{color:var(--blue)}
  .welcome-description{max-width:485px;color:var(--muted);font-size:1rem;line-height:1.7}
  .primary-action{height:48px;margin-top:27px;padding:0 17px 0 20px;display:inline-flex;align-items:center;gap:20px;background:#dce8ff;color:#101722;border:0;border-radius:8px;font-size:.88rem;font-weight:650;font-family:inherit;cursor:pointer;transition:background .2s}
  .primary-action:hover{background:white}.primary-action svg{width:16px;height:16px;fill:none;stroke:currentColor;stroke-width:1.7}
  .workflow-card{width:100%;max-width:460px;justify-self:end;display:flex;flex-direction:column;text-align:left;padding:20px 23px 14px;border:1px solid var(--line);border-radius:13px;color:var(--ink);background:#10131a;cursor:pointer}
  .workflow-header{display:flex;justify-content:space-between;padding-bottom:13px;border-bottom:1px solid var(--line);color:#8b95a7;font-size:.64rem;font-weight:650;letter-spacing:.13em}
  .workflow-step{display:flex;align-items:flex-start;gap:15px;padding:16px 2px;border-bottom:1px solid rgba(182,191,215,.09)}
  .workflow-step>span{padding-top:2px;color:#8db4ff;font-size:.66rem;font-weight:650;letter-spacing:.08em}
  .workflow-step div{display:flex;flex-direction:column;gap:3px}.workflow-step strong{font-size:.88rem;font-weight:600}.workflow-step small{font-size:.76rem;color:#8c94a2}
  .workflow-link{display:flex;justify-content:space-between;padding:14px 2px 0;color:#b7ccf5;font-size:.78rem;font-weight:600}
  .metrics-section{max-width:1180px;margin:0 auto}.section-heading{display:flex;align-items:flex-end;justify-content:space-between;gap:20px;margin-bottom:14px}.section-heading h2{font-family:inherit;font-size:1.25rem;font-weight:600;letter-spacing:-.025em;margin-top:6px}.source-note{font-size:.73rem;color:#7f8795}
  .metrics-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.metric-card{min-height:112px;padding:15px 18px;border:1px solid var(--line);border-radius:10px;background:rgba(17,20,28,.78);display:flex;flex-direction:column;align-items:flex-start}.metric-label{font-size:.75rem;color:#a0a7b3}.metric-card strong{font-size:1.6rem;letter-spacing:-.04em;font-weight:650;margin-top:4px}.metric-detail{font-size:.68rem;color:#788193}
  .overview-footer{max-width:1180px;margin:32px auto 0;border-top:1px solid var(--line);padding:13px 0 0;display:flex;justify-content:space-between;gap:16px;color:#778091;font-family:inherit;font-size:.66rem}
  @media(max-width:800px){.overview-page{padding-top:78px}.welcome-section{grid-template-columns:1fr;padding:32px 0 38px;gap:30px}.workflow-card{max-width:none}.section-heading{align-items:flex-start;flex-direction:column;gap:5px}}
  @media(max-width:520px){.welcome-copy h1{font-size:2.8rem;letter-spacing:-.045em}.welcome-description{font-size:.92rem}.workflow-card{padding:17px}.metrics-grid{grid-template-columns:1fr}.overview-footer{margin-top:25px}}
</style>

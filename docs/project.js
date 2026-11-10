// Drive A2C formatters for the generic site (ml-project-site skill).
const sc = (x) => (x == null || x === '' || isNaN(x)) ? '–' : (+x).toFixed(1);
const at = (e) => e.step != null ? ` at ${(e.step / 1e6).toFixed(2)} M steps` : '';
const chkText = (c) => c ? `mean score ${sc(c.score)} on ${c.n} validation tracks; laps finished ${pct(c.laps, 0)}, track covered ${pct(c.tiles, 0)}, worst track ${sc(c.worst)}` : '';
window.worldText = (w) => (typeof w === 'string' ? w : kv(w));
window.rewardText = (r) => (typeof r === 'string' ? r : kv(r));
window.passText = (p) => (typeof p === 'string' ? p : kv(p));
window.eventText = (e) => {
  switch (e.kind) {
    case 'run_start': return [`Run ${e.run} started`, `${(e.algo || '').toUpperCase()}, ${e.num_envs} cars driving in parallel, ${(e.total_steps / 1e6).toFixed(0)} M steps planned, ${e.obs_dim} numbers in.`];
    case 'val_check': return [`Validation${at(e)}: ${sc(e.check.score)}`, chkText(e.check)];
    case 'goal_best': return [`New best validation score${at(e)}: ${sc(e.check.score)}`, chkText(e.check) + "\nA best-of-many check flatters (winner's curse); the goal is measured on 100 other tracks."];
    case 'stage_pass': return [`✅ Stage passed: ${e.stage}${at(e)}`, chkText(e.check)];
    case 'stage_cap': return [`Stage "${e.stage}" reached its step cap${at(e)}; moving on`, chkText(e.check)];
    case 'stage_start': return [`Stage started: ${e.stage}${at(e)}`, e.expect ? 'Expected: ' + e.expect : ''];
    case 'goal_check': return [`🎯 Goal check: mean score ${sc(e.mean)} on ${e.n} unseen tracks`, `${e.label || ''}. 95% interval of the mean ${e.ci ? sc(e.ci[0]) + ' – ' + sc(e.ci[1]) : '–'}. ${e.note || ''}`];
    case 'run_end': return [`Run ${e.run} finished${at(e)}`, `Best validation score ${sc(e.best_val)}.`];
    case 'stopped': return [`Run ${e.run} stopped${at(e)} (checkpoint saved)`, ''];
    case 'resume': return [`Run ${e.run} resumed${at(e)}`, ''];
  }
  return null;
};

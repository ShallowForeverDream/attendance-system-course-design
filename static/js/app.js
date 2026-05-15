const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
let me = null;
let mediaStream = null;
let livenessFlashOverlay = null;

async function api(url, options = {}) {
  const res = await fetch(url, {
    credentials: 'same-origin',
    headers: options.body instanceof FormData ? {} : {'Content-Type': 'application/json'},
    ...options,
  });
  const type = res.headers.get('content-type') || '';
  const data = type.includes('application/json') ? await res.json() : await res.text();
  if (!res.ok || (data && data.ok === false)) {
    throw new Error((data && data.error) || res.statusText || '请求失败');
  }
  return data;
}

function show(el, visible) { el.classList.toggle('hidden', !visible); }
function fmt(n) { return Number(n || 0).toFixed(3); }
function escapeHtml(s) { return String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function groupStudentLabel(r) { return `${r.student_no || ''} ${r.name || ''}`.trim(); }
function emotionText(emotion) {
  if (!emotion) return 'unknown（0.000，engine=unknown）';
  const engine = emotion.engine || 'unknown';
  const model = emotion.model ? `，model=${emotion.model}` : '';
  const guarded = emotion.model_emotion ? `，原始=${emotion.model_emotion}/${fmt(emotion.model_confidence)}` : '';
  return `${emotion.emotion || 'unknown'}（${fmt(emotion.confidence)}，engine=${engine}${model}${guarded}）`;
}
function openBulkFaceImportOverlay(res) {
  const overlay = $('#bulkFaceImportOverlay');
  const box = $('#bulkFaceImportResult');
  const successItems = (res.added || []).map(x => `<li class="good">${escapeHtml(x.file)} → ${escapeHtml(x.student_no)} ${escapeHtml(x.name)} · 质量 ${fmt(x.quality)}</li>`).join('');
  const errorItems = (res.errors || []).map(x => `<li class="bad">${escapeHtml(x.file || '-')}${x.student_no ? ` → ${escapeHtml(x.student_no)} ${escapeHtml(x.name || '')}` : ''}：${escapeHtml(x.error)}</li>`).join('');
  box.innerHTML = `
    <div class="import-summary">
      <div class="card"><div class="num">${res.students_added || 0}</div><div class="label">新增学生</div></div>
      <div class="card"><div class="num">${res.students_updated || 0}</div><div class="label">更新学生</div></div>
      <div class="card"><div class="num">${res.samples_added || 0}</div><div class="label">新增样本</div></div>
    </div>
    <div class="import-grid">
      <div class="import-list">
        <h4>成功清单（${(res.added || []).length}）</h4>
        ${successItems ? `<ul>${successItems}</ul>` : '<p class="small">无成功记录</p>'}
      </div>
      <div class="import-list">
        <h4>失败清单（${(res.errors || []).length}）</h4>
        ${errorItems ? `<ul>${errorItems}</ul>` : '<p class="small">无失败记录</p>'}
      </div>
    </div>`;
  show(overlay, true);
  overlay.setAttribute('aria-hidden', 'false');
}
function closeBulkFaceImportOverlay() {
  const overlay = $('#bulkFaceImportOverlay');
  if (!overlay) return;
  show(overlay, false);
  overlay.setAttribute('aria-hidden', 'true');
}

async function refreshMe() {
  const data = await api('/api/me');
  me = data.user;
  show($('#loginPanel'), !me);
  show($('#appPanel'), !!me);
  show($('#logoutBtn'), !!me);
  $('#currentUser').textContent = me ? `${me.username}（${me.role === 'teacher' ? '教师' : '学生'}）` : '未登录';
  $$('[data-teacher]').forEach(el => show(el, me?.role === 'teacher'));
  if (me) await Promise.allSettled([loadSummary(), loadRecords(), loadStudents(), loadStats()]);
}

$('#loginForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  $('#loginMsg').textContent = '';
  const form = new FormData(e.currentTarget);
  try {
    await api('/api/login', {method: 'POST', body: JSON.stringify(Object.fromEntries(form))});
    await refreshMe();
  } catch (err) { $('#loginMsg').textContent = err.message; }
});

$('#logoutBtn').addEventListener('click', async () => {
  await api('/api/logout', {method: 'POST', body: '{}'});
  location.reload();
});

$$('.tabs button').forEach(btn => btn.addEventListener('click', async () => {
  $$('.tabs button').forEach(b => b.classList.remove('active'));
  $$('.tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  $('#' + btn.dataset.tab).classList.add('active');
  if (btn.dataset.tab === 'records') await loadRecords();
  if (btn.dataset.tab === 'students') await loadStudents();
  if (btn.dataset.tab === 'stats') await loadStats();
  if (btn.dataset.tab === 'scorecard') await loadScorecard();
}));

async function loadSummary() {
  const data = await api('/api/summary');
  const labels = {students: '学生数', face_samples: '人脸样本', attendance: '考勤记录', activities: '活动次数'};
  $('#summaryCards').innerHTML = Object.entries(data.counts)
    .map(([k,v]) => `<div class="card"><div class="num">${v}</div><div class="label">${labels[k] || k}</div></div>`).join('');
}

async function startCamera() {
  if (mediaStream) return;
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: {ideal: 960},
        height: {ideal: 720},
        frameRate: {ideal: 30, max: 60},
        facingMode: 'user'
      },
      audio: false
    });
    $('#video').srcObject = mediaStream;
    $('#attendanceMsg').className = 'result-box';
    $('#attendanceMsg').textContent = '摄像头已开启，可开始活体打卡。';
  } catch (err) {
    $('#attendanceMsg').className = 'result-box bad';
    $('#attendanceMsg').textContent = '摄像头调用失败：' + err.message + '\n请确认浏览器权限、HTTPS/localhost 环境。';
  }
}
$('#startCameraBtn').addEventListener('click', startCamera);

function ensureFlashOverlay() {
  if (livenessFlashOverlay) return livenessFlashOverlay;
  const card = $('.camera-card');
  livenessFlashOverlay = document.createElement('div');
  livenessFlashOverlay.id = 'livenessFlashOverlay';
  livenessFlashOverlay.className = 'liveness-flash hidden';
  livenessFlashOverlay.innerHTML = '<div class="flash-core">LIVE CHALLENGE</div>';
  card?.appendChild(livenessFlashOverlay);
  return livenessFlashOverlay;
}

function setFlashColor(rgb, label = '') {
  const overlay = ensureFlashOverlay();
  if (!rgb) {
    overlay.classList.add('hidden');
    return;
  }
  const color = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
  overlay.style.background = color;
  overlay.style.boxShadow = `0 0 68px 18px ${color}, inset 0 0 80px 20px ${color}`;
  overlay.querySelector('.flash-core').textContent = label || `FLASH ${rgb.join(',')}`;
  overlay.classList.remove('hidden');
}

function hideFlashColor() {
  if (livenessFlashOverlay) livenessFlashOverlay.classList.add('hidden');
}

function activeFlashForStep(step, elapsed) {
  const seq = step.flash_sequence || [];
  if (!seq.length) return null;
  const interval = step.flash_interval_ms || 700;
  const index = Math.floor(elapsed / interval) % seq.length;
  const item = seq[index];
  return {index, rgb: item.rgb || item, name: item.name || String(index + 1)};
}

function captureFrame(stage, extra = {}) {
  const video = $('#video');
  const canvas = $('#captureCanvas');
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  // 将随机闪光色块写入采集帧边缘，帮助普通摄像头捕获当前挑战光照；
  // 后端仍以 session 中的随机序列校验，前端元数据不能单独绕过。
  if (extra.flash_rgb) {
    const [r, g, b] = extra.flash_rgb;
    ctx.save();
    ctx.globalAlpha = 0.42;
    ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
    const band = Math.max(34, Math.round(canvas.width * 0.075));
    ctx.fillRect(0, 0, canvas.width, band);
    ctx.fillRect(0, canvas.height - band, canvas.width, band);
    ctx.fillRect(0, 0, band, canvas.height);
    ctx.fillRect(canvas.width - band, 0, band, canvas.height);
    ctx.restore();
  }
  return {stage, image: canvas.toDataURL('image/jpeg', 0.82), ...extra};
}

$('#manualCaptureBtn')?.addEventListener('click', async () => {
  await startCamera();
  $('#attendanceMsg').className = 'result-box';
  $('#attendanceMsg').textContent = '已手动抓拍一帧，正在做质量、人脸库和情绪预检...';
  try {
    const frame = captureFrame(0);
    const data = await api('/api/attendance/preview', {method: 'POST', body: JSON.stringify({image: frame.image})});
    const p = data.preview;
    $('#attendanceMsg').className = 'result-box ok';
    $('#attendanceMsg').textContent =
      `手动抓拍预检完成（不写入考勤）\n` +
      `检测人脸数：${p.face_count}，质量分：${fmt(p.quality)}\n` +
      `最佳匹配：${p.matched ? `${p.student_no} ${p.name}` : '未达到正式阈值'}\n` +
      `学生聚合分：${fmt(p.score)}（阈值 ${fmt(p.threshold)}），最佳单样本：${fmt(p.best_sample_score)}，该生样本数：${p.sample_count || 0}\n` +
      `第二名：${fmt(p.second_score)}，间隔：${fmt(p.score_margin)}\n` +
      `情绪：${emotionText(p.emotion)}\n` +
      `说明：${p.note}`;
  } catch (err) {
    $('#attendanceMsg').className = 'result-box bad';
    $('#attendanceMsg').textContent = err.message;
  }
});

function wait(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }
function actionGuide(action) {
  const guides = {
    move_left: '保持脸在画面内，平移到屏幕左侧。',
    move_right: '保持脸在画面内，平移到屏幕右侧。',
    move_closer: '脸部慢慢靠近摄像头，让人脸框变大。',
    move_away: '脸部慢慢远离摄像头，让人脸框变小。',
    nod: '轻轻点头一次，动作幅度要明显但不要离开画面。',
    blink: '看着摄像头，脸保持稳定，完成一次“睁眼-闭眼约0.2秒-再睁眼”。',
    open_mouth: '保持正脸，张嘴一次后闭合。',
    turn_left: '保持脸在画面内，向左转头一次。',
    turn_right: '保持脸在画面内，向右转头一次。',
    flash_response: '保持正脸不做其它动作，尽量靠近屏幕并调高屏幕亮度，让随机颜色照到脸上。'
  };
  return guides[action] || '按屏幕提示完成动作。';
}

async function captureStageUntilPass(challenge, step, allFrames) {
  const timeoutMs = (step.timeout_seconds || challenge.group_timeout_seconds || 5) * 1000;
  const startedAt = performance.now();
  const frames = [];
  let lastCheckAt = 0;
  let lastReason = '等待动作变化...';
  const captureIntervalMs = step.action === 'blink' ? 70 : (step.action === 'flash_response' ? 120 : 140);
  const checkIntervalMs = step.action === 'blink' ? 280 : 480;
  const minCheckFrames = step.action === 'blink' ? 6 : 5;

  while (performance.now() - startedAt < timeoutMs) {
    const elapsed = performance.now() - startedAt;
    const flash = activeFlashForStep(step, elapsed);
    if (flash) setFlashColor(flash.rgb, `STAGE ${step.group || step.stage} · ${flash.name}`);
    const frame = captureFrame(step.stage, {
      action: step.action,
      stage_elapsed_ms: Math.round(elapsed),
      captured_at_ms: Date.now(),
      flash_index: flash ? flash.index : null,
      flash_rgb: flash ? flash.rgb : null
    });
    frames.push(frame);
    allFrames.push(frame);

    const remain = Math.max(0, Math.ceil((timeoutMs - elapsed) / 1000));
    $('#attendanceMsg').textContent =
      `${step.hint || `第 ${step.group}/${challenge.group_count} 组`}\n` +
      `动作：${step.label}\n提示：${actionGuide(step.action)}\n` +
      `随机闪光：${flash ? flash.name + ' [' + flash.rgb.join(',') + ']' : '未启用'}\n` +
      `采样：${frames.length} 帧 / ${captureIntervalMs}ms\n剩余：${remain} 秒\n状态：${lastReason}`;

    if (frames.length >= minCheckFrames && performance.now() - lastCheckAt > checkIntervalMs) {
      lastCheckAt = performance.now();
      try {
        const data = await api('/api/attendance/liveness-stage', {
          method: 'POST',
          body: JSON.stringify({
            challenge_id: challenge.id,
            stage: step.stage,
            elapsed_ms: Math.round(performance.now() - startedAt),
            frames
          })
        });
        lastReason = data.reason || '正在检测动作...';
        if (data.stage_pass) {
          return {ok: true, frames, reason: data.reason};
        }
      } catch (err) {
        lastReason = err.message;
      }
    }
    await wait(captureIntervalMs);
  }
  return {ok: false, frames, reason: `第 ${step.group || step.stage} 组动作未在 ${Math.round(timeoutMs / 1000)} 秒内完成：${lastReason}`};
}

$('#startCheckBtn').addEventListener('click', async () => {
  await startCamera();
  $('#attendanceMsg').className = 'result-box';
  try {
    $('#challengeSteps').innerHTML = '';
    const challenge = (await api('/api/attendance/challenge')).challenge;
    const stepsEl = $('#challengeSteps');
    stepsEl.innerHTML = challenge.steps.map(s =>
      `<li data-stage="${s.stage}"><strong>第 ${s.group}/${challenge.group_count} 组</strong>：${escapeHtml(s.label)} <span class="small">≤${s.timeout_seconds}s · 随机闪光${(s.flash_sequence || []).length}段</span></li>`
    ).join('');
    const frames = [];
    for (const step of challenge.steps) {
      $$('li', stepsEl).forEach(li => li.classList.toggle('active', Number(li.dataset.stage) === step.stage));
      $('#attendanceMsg').textContent = `${step.hint}\n动作：${step.label}\n提示：${actionGuide(step.action)}`;
      await wait(300);
      const stageResult = await captureStageUntilPass(challenge, step, frames);
      if (!stageResult.ok) {
        $(`li[data-stage="${step.stage}"]`, stepsEl)?.classList.add('failed');
        throw new Error(stageResult.reason + '\n请重新开始活体打卡。');
      }
      $(`li[data-stage="${step.stage}"]`, stepsEl)?.classList.add('done');
      $('#attendanceMsg').textContent = `第 ${step.group}/${challenge.group_count} 组已通过：${step.label}\n准备进入下一组...`;
      await wait(450);
    }
    hideFlashColor();
    $('#attendanceMsg').textContent = '采集完成，正在后端进行活体检测、人脸比对与情绪分析...';
    const data = await api('/api/attendance/check', {method: 'POST', body: JSON.stringify({challenge_id: challenge.id, frames})});
    const r = data.result;
    $('#attendanceMsg').className = 'result-box ' + (r.status === 'success' ? 'ok' : 'bad');
    $('#attendanceMsg').textContent =
      `考勤状态：${r.status === 'success' ? '成功' : '失败'}\n` +
      `姓名：${r.name || '-'}\n学号：${r.student_no || '-'}\n时间：${r.time}\n` +
      `活体：${r.liveness.pass ? '通过' : '未通过'}（${r.liveness.reason}，分数 ${fmt(r.liveness.score)}）\n` +
      `人脸匹配分：${fmt(r.face_score)}（阈值 ${fmt(r.face_threshold)}，最佳单样本 ${fmt(r.best_sample_score)}，样本数 ${r.sample_count || 0}）\n` +
      `情绪：${emotionText(r.emotion)}\n备注：${r.note || '-'}`;
    await Promise.allSettled([loadSummary(), loadRecords(), loadStats()]);
  } catch (err) {
    hideFlashColor();
    $('#attendanceMsg').className = 'result-box bad';
    $('#attendanceMsg').textContent = err.message;
  }
});

$('#recordFilter').addEventListener('submit', async (e) => { e.preventDefault(); await loadRecords(); });
async function loadRecords() {
  if (!me) return;
  const params = new URLSearchParams(new FormData($('#recordFilter')));
  const data = await api('/api/attendance?' + params.toString());
  $('#recordTable').innerHTML =
    `<thead><tr><th>时间</th><th>学号</th><th>姓名</th><th>状态</th><th>活体</th><th>人脸分</th><th>情绪</th><th>备注</th></tr></thead><tbody>` +
    data.records.map(r => `<tr><td>${r.captured_at}</td><td>${escapeHtml(r.student_no)}</td><td>${escapeHtml(r.name)}</td><td><span class="badge ${r.status}">${r.status}</span></td><td>${r.liveness_pass ? '通过' : '失败'} / ${fmt(r.liveness_score)}</td><td>${fmt(r.face_score)}</td><td>${escapeHtml(r.emotion)}</td><td>${escapeHtml(r.note)}</td></tr>`).join('') +
    `</tbody>`;
}

$('#studentForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const body = Object.fromEntries(new FormData(e.currentTarget));
  try {
    await api('/api/students', {method: 'POST', body: JSON.stringify(body)});
    e.currentTarget.reset();
    await Promise.all([loadStudents(), loadSummary()]);
  } catch (err) { alert(err.message); }
});
$('#reloadStudentsBtn').addEventListener('click', loadStudents);
$('#studentSearch').addEventListener('keydown', e => { if (e.key === 'Enter') loadStudents(); });

async function loadStudents() {
  if (!me) return;
  const q = $('#studentSearch')?.value || '';
  const data = await api('/api/students?q=' + encodeURIComponent(q));
  const canEdit = me.role === 'teacher';
  const cameraSelect = $('#cameraStudentSelect');
  if (cameraSelect) {
    cameraSelect.innerHTML = data.students.map(s => `<option value="${s.id}">${escapeHtml(s.student_no)} ${escapeHtml(s.name)}</option>`).join('');
  }
  $('#studentTable').innerHTML =
    `<thead><tr><th>学号</th><th>姓名</th><th>班级</th><th>状态</th><th>样本数</th>${canEdit ? '<th>人脸样本上传</th><th>操作</th>' : ''}</tr></thead><tbody>` +
    data.students.map(s => `<tr data-id="${s.id}"><td>${escapeHtml(s.student_no)}</td><td>${escapeHtml(s.name)}</td><td>${escapeHtml(s.class_name)}</td><td>${escapeHtml(s.status)}</td><td>${s.face_count}</td>${canEdit ? `<td><form class="inline-upload"><input type="file" name="faces" accept="image/*" multiple required /><button>上传样本</button></form></td><td><button class="ghost view-faces" type="button">样本</button><button class="ghost edit-student" type="button">编辑</button><button class="ghost delete-student" type="button">删除</button></td>` : ''}</tr>`).join('') +
    `</tbody>`;
  $$('.inline-upload', $('#studentTable')).forEach(form => form.addEventListener('submit', async e => {
    e.preventDefault();
    const tr = e.currentTarget.closest('tr');
    try {
      const res = await api(`/api/students/${tr.dataset.id}/faces`, {method: 'POST', body: new FormData(e.currentTarget)});
      alert(`上传完成：成功 ${res.added.length} 张，失败 ${res.errors.length} 张` + (res.errors.length ? '\n' + res.errors.map(x => `${x.file}: ${x.error}`).join('\n') : ''));
      await Promise.all([loadStudents(), loadSummary()]);
    } catch (err) { alert(err.message); }
  }));
  $$('.delete-student', $('#studentTable')).forEach(btn => btn.addEventListener('click', async e => {
    const tr = e.currentTarget.closest('tr');
    if (!confirm('确认删除该学生及其人脸样本？')) return;
    await api(`/api/students/${tr.dataset.id}`, {method: 'DELETE'});
    await Promise.all([loadStudents(), loadSummary()]);
  }));
  $$('.edit-student', $('#studentTable')).forEach(btn => btn.addEventListener('click', async e => {
    const tr = e.currentTarget.closest('tr');
    const s = data.students.find(x => String(x.id) === String(tr.dataset.id));
    if (!s) return;
    const student_no = prompt('修改学号：', s.student_no);
    if (student_no === null) return;
    const name = prompt('修改姓名：', s.name);
    if (name === null) return;
    const class_name = prompt('修改班级/专业：', s.class_name || '');
    if (class_name === null) return;
    const gender = prompt('修改性别：', s.gender || '');
    if (gender === null) return;
    const status = prompt('修改状态（active/inactive）：', s.status || 'active');
    if (status === null) return;
    try {
      await api(`/api/students/${tr.dataset.id}`, {method: 'PUT', body: JSON.stringify({student_no, name, class_name, gender, status})});
      await Promise.all([loadStudents(), loadSummary(), loadScorecard()]);
    } catch (err) { alert(err.message); }
  }));
  $$('.view-faces', $('#studentTable')).forEach(btn => btn.addEventListener('click', async e => {
    const tr = e.currentTarget.closest('tr');
    await showFaceSamples(tr.dataset.id);
  }));
}

async function showFaceSamples(studentId) {
  const panel = $('#faceSamplesPanel');
  panel.classList.remove('hidden');
  panel.innerHTML = '<div class="notice">正在加载人脸样本...</div>';
  try {
    const data = await api(`/api/students/${studentId}/faces`);
    const title = `${data.student.student_no} ${data.student.name}`;
    panel.innerHTML =
      `<div class="section-head compact"><div><h3>人脸样本：${escapeHtml(title)}</h3><p>可单独删除质量差、重复或误导入的人脸样本，满足人脸库“单个添加/删除/修改维护”的现场展示。</p></div><button id="closeFacesPanel" class="ghost" type="button">关闭</button></div>` +
      (data.faces.length ? `<div class="face-sample-grid">` + data.faces.map(f =>
        `<div class="face-sample-card" data-face-id="${f.id}"><a href="${f.url}" target="_blank" title="打开原图"><img src="${f.url}" alt="face sample ${f.id}" loading="lazy" onerror="this.closest('.face-sample-card').classList.add('broken'); this.replaceWith(Object.assign(document.createElement('div'), {className:'broken-img', textContent:'图片文件不存在或路径失效'}));" /></a><div class="small">ID ${f.id} · 质量 ${fmt(f.quality)}<br>${escapeHtml(f.created_at)}${f.exists === false ? '<br><span class="bad-text">文件缺失</span>' : ''}</div><button class="ghost delete-face" type="button">删除该样本</button></div>`
      ).join('') + `</div>` : '<p class="small">该学生暂无人脸样本，可通过文件上传或摄像头补采添加。</p>');
    $('#closeFacesPanel').addEventListener('click', () => panel.classList.add('hidden'));
    $$('.delete-face', panel).forEach(btn => btn.addEventListener('click', async e => {
      const card = e.currentTarget.closest('.face-sample-card');
      if (!confirm('确认删除该人脸样本？学生信息不会被删除。')) return;
      await api(`/api/faces/${card.dataset.faceId}`, {method: 'DELETE'});
      await Promise.all([showFaceSamples(studentId), loadStudents(), loadSummary(), loadScorecard()]);
    }));
  } catch (err) {
    panel.innerHTML = `<div class="result-box bad">${escapeHtml(err.message)}</div>`;
  }
}

$('#bulkStudentsBtn')?.addEventListener('click', async () => {
  try {
    const students = JSON.parse($('#bulkStudentsInput').value || '[]');
    const res = await api('/api/students/bulk', {method: 'POST', body: JSON.stringify({students})});
    alert(`批量导入完成：新增 ${res.added}，更新 ${res.updated}，错误 ${res.errors.length}`);
    await Promise.all([loadStudents(), loadSummary(), loadScorecard()]);
  } catch (err) { alert('批量导入失败：' + err.message); }
});

$('#bulkFaceImportForm')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const msg = $('#bulkFaceImportMsg');
  msg.textContent = '正在批量导入人脸图片，请稍候...';
  try {
    const res = await api('/api/students/faces/bulk', {method: 'POST', body: new FormData(e.currentTarget)});
    msg.textContent = `导入完成：新增学生 ${res.students_added}，更新学生 ${res.students_updated}，新增样本 ${res.samples_added}，失败 ${res.errors.length}`;
    openBulkFaceImportOverlay(res);
    await Promise.all([loadStudents(), loadSummary(), loadScorecard()]);
  } catch (err) {
    msg.textContent = '批量导入失败：' + err.message;
  }
});

$('#closeBulkFaceImportOverlay')?.addEventListener('click', closeBulkFaceImportOverlay);
$('#bulkFaceImportOverlay')?.addEventListener('click', e => {
  if (e.target === e.currentTarget) closeBulkFaceImportOverlay();
});

$('#addFaceFromCameraBtn')?.addEventListener('click', async () => {
  await startCamera();
  const studentId = $('#cameraStudentSelect').value;
  if (!studentId) return alert('请先选择学生');
  try {
    const frame = captureFrame(0);
    const res = await api(`/api/students/${studentId}/face-from-camera`, {method: 'POST', body: JSON.stringify({image: frame.image})});
    alert(`摄像头人脸样本已入库，质量分：${fmt(res.quality)}`);
    await Promise.all([loadStudents(), loadSummary(), loadScorecard()]);
  } catch (err) { alert(err.message); }
});

$('#groupForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  $('#groupResult').innerHTML = '<div class="notice">正在识别合照，请稍候...</div>';
  try {
    const data = await api('/api/group/recognize', {method: 'POST', body: new FormData(e.currentTarget)});
    const autoIds = new Set(data.results.filter(r => r.matched && r.student_id).map(r => String(r.student_id)));
    const candidateRows = data.results.filter(r => !r.matched && r.candidate_student_id);
    $('#groupResult').dataset.activityId = data.activity_id;
    $('#groupResult').innerHTML =
      `<div class="notice"><strong>检测到 ${data.faces_detected} 张人脸，自动匹配 ${data.matched_count} 名学生，待人工确认 ${data.review_count || 0} 张。</strong><br>` +
      `真实考勤系统建议采用“自动识别 + 教师确认”闭环，确认后会重写活动名单和频次统计。</div>` +
      `<img src="${data.annotated_url}" alt="合照识别标注结果" />` +
      `<div class="group-confirm-panel"><h3>最终参与名单确认</h3>` +
      `<div class="small">自动匹配项已默认勾选；低置信候选可人工勾选，缺漏人员也可在下面补选。</div>` +
      `<div id="groupChecklist" class="check-grid"></div>` +
      `<div class="toolbar"><select id="manualStudentSelect"></select><button id="manualAddBtn" type="button">补选学生</button><button id="confirmGroupBtn" type="button">确认并写入活动频次</button></div>` +
      `<div id="confirmGroupMsg" class="small"></div></div>` +
      `<div class="table-wrap"><table><thead><tr><th>序号</th><th>自动结果</th><th>最佳候选</th><th>状态</th><th>聚合分</th><th>单样本/样本数</th><th>间隔</th><th>情绪</th><th>情绪引擎</th></tr></thead><tbody>` +
      data.results.map(r => `<tr><td>${r.face_index}</td><td>${escapeHtml(r.matched ? groupStudentLabel(r) : '未自动确认')}</td><td>${escapeHtml(r.candidate_student_no ? `${r.candidate_student_no} ${r.candidate_name}` : '-')}</td><td>${r.matched ? '<span class="badge ok">自动确认</span>' : (r.needs_review ? '<span class="badge warn">待确认</span>' : '<span class="badge bad">未匹配</span>')}</td><td>${fmt(r.score)}</td><td>${fmt(r.best_sample_score)} / ${r.sample_count || 0}</td><td>${fmt(r.score_margin)}</td><td>${escapeHtml(r.emotion)}（${fmt(r.emotion_confidence)}）</td><td>${escapeHtml(r.emotion_engine || 'unknown')}</td></tr>`).join('') +
      `</tbody></table></div>`;
    const checklist = $('#groupChecklist');
    const addCheck = (id, label, checked, source) => {
      if (!id || $(`#groupChecklist input[value="${id}"]`)) return;
      checklist.insertAdjacentHTML('beforeend', `<label class="check-item"><input type="checkbox" value="${id}" ${checked ? 'checked' : ''}/> <span>${escapeHtml(label)}</span><em>${escapeHtml(source)}</em></label>`);
    };
    data.results.filter(r => r.matched).forEach(r => addCheck(r.student_id, groupStudentLabel(r), true, '自动'));
    candidateRows.forEach(r => addCheck(r.candidate_student_id, `${r.candidate_student_no} ${r.candidate_name}`, false, '候选'));
    const students = await api('/api/students');
    $('#manualStudentSelect').innerHTML = students.students.map(s => `<option value="${s.id}">${escapeHtml(s.student_no)} ${escapeHtml(s.name)}</option>`).join('');
    $('#manualAddBtn').addEventListener('click', () => {
      const opt = $('#manualStudentSelect').selectedOptions[0];
      if (opt) addCheck(opt.value, opt.textContent, true, '补选');
    });
    $('#confirmGroupBtn').addEventListener('click', async () => {
      const ids = $$('#groupChecklist input:checked').map(x => Number(x.value));
      const res = await api(`/api/group/${data.activity_id}/participants`, {method: 'POST', body: JSON.stringify({student_ids: ids})});
      $('#confirmGroupMsg').textContent = `已确认 ${res.count} 名参与者，活动频次/情绪统计已更新。`;
      await Promise.allSettled([loadSummary(), loadStats(), loadScorecard()]);
    });
    if (!checklist.children.length) checklist.innerHTML = '<p class="small">暂无自动名单，请通过“补选学生”添加最终参与者。</p>';
    await Promise.allSettled([loadSummary(), loadStats()]);
  } catch (err) {
    $('#groupResult').innerHTML = `<div class="result-box bad">${escapeHtml(err.message)}</div>`;
  }
});

async function loadStats() {
  if (!me) return;
  const [activity, emotion] = await Promise.all([api('/api/group/stats'), api('/api/emotions/stats')]);
  renderBars($('#activityStats'), activity.stats, 'name', 'count', r => `${r.student_no || ''} ${r.name || ''}`.trim() || '无');
  renderBars($('#emotionStats'), emotion.stats, 'emotion', 'count', r => `${r.scene}/${r.emotion}`);
}

function renderBars(root, rows, labelKey, valueKey, labelFn) {
  if (!rows.length) { root.innerHTML = '<p class="small">暂无统计数据</p>'; return; }
  const max = Math.max(...rows.map(r => Number(r[valueKey] || 0)), 1);
  root.innerHTML = rows.map(r => {
    const val = Number(r[valueKey] || 0);
    return `<div class="bar-row"><div>${escapeHtml(labelFn ? labelFn(r) : r[labelKey])}</div><div class="bar"><span style="width:${Math.max(4, val / max * 100)}%"></span></div><strong>${val}</strong></div>`;
  }).join('');
}

$('#securityCameraBtn')?.addEventListener('click', startCamera);
$('#staticAttackBtn')?.addEventListener('click', async () => {
  await startCamera();
  $('#securityResult').className = 'result-box';
  $('#securityResult').textContent = '正在模拟静态照片/重复帧攻击...';
  try {
    const frame = captureFrame(0);
    const data = await api('/api/liveness/self-test', {method: 'POST', body: JSON.stringify({image: frame.image})});
    const live = data.liveness;
    $('#securityResult').className = 'result-box ' + (!live.pass ? 'ok' : 'bad');
    $('#securityResult').textContent =
      `攻击类型：${data.attack}\n预期：${data.expected}\n实际活体：${live.pass ? '通过（需要改进）' : '拒绝（符合预期）'}\n` +
      `原因：${live.reason}\n分数：${fmt(live.score)}\n重复帧比例：${fmt(1 - (live.unique_frame_ratio || 0))}\n平均帧差：${live.avg_frame_diff}`;
  } catch (err) {
    $('#securityResult').className = 'result-box bad';
    $('#securityResult').textContent = err.message;
  }
});

$('#sampleAttackBtn')?.addEventListener('click', async () => {
  $('#securityResult').className = 'result-box';
  $('#securityResult').textContent = '正在使用人脸库样本模拟静态照片/重复帧攻击...';
  try {
    const data = await api('/api/liveness/self-test-sample', {method: 'POST', body: '{}'});
    const live = data.liveness;
    $('#securityResult').className = 'result-box ' + (!live.pass ? 'ok' : 'bad');
    $('#securityResult').textContent =
      `攻击类型：${data.attack}\n样本：${data.sample.student_no} ${data.sample.name}\n预期：${data.expected}\n实际活体：${live.pass ? '通过（需要改进）' : '拒绝（符合预期）'}\n` +
      `原因：${live.reason}\n分数：${fmt(live.score)}\n重复帧比例：${fmt(1 - (live.unique_frame_ratio || 0))}\n平均帧差：${live.avg_frame_diff}`;
  } catch (err) {
    $('#securityResult').className = 'result-box bad';
    $('#securityResult').textContent = err.message;
  }
});

$('#movingPhotoAttackBtn')?.addEventListener('click', async () => {
  $('#securityResult').className = 'result-box';
  $('#securityResult').textContent = '正在模拟“举着照片移动”攻击...';
  try {
    const data = await api('/api/liveness/self-test-moving-photo', {method: 'POST', body: '{}'});
    const live = data.liveness;
    $('#securityResult').className = 'result-box ' + (!live.pass ? 'ok' : 'bad');
    $('#securityResult').textContent =
      `攻击类型：${data.attack}
样本：${data.sample.student_no} ${data.sample.name}
预期：${data.expected}
实际活体：${live.pass ? '通过（需要改进）' : '拒绝（符合预期）'}
` +
      `原因：${live.reason}
分数：${fmt(live.score)}
动作通过：${live.action_pass_count}/${live.required_action_count}
抗平面伪造：${live.anti_spoof_pass ? '通过' : '拒绝'}
归一化人脸帧差：${live.avg_crop_diff}
刚性平面判定：${live.rigid_planar_replay ? '是' : '否'}`;
  } catch (err) {
    $('#securityResult').className = 'result-box bad';
    $('#securityResult').textContent = err.message;
  }
});

$('#prerecordedAttackBtn')?.addEventListener('click', async () => {
  $('#securityResult').className = 'result-box';
  $('#securityResult').textContent = '正在模拟“预录视频无实时闪光响应”攻击...';
  try {
    const data = await api('/api/liveness/self-test-prerecorded', {method: 'POST', body: '{}'});
    const live = data.liveness;
    const flash = live.flash_challenge || {};
    $('#securityResult').className = 'result-box ' + (!live.pass ? 'ok' : 'bad');
    $('#securityResult').textContent =
      `攻击类型：${data.attack}
样本：${data.sample.student_no} ${data.sample.name}
预期：${data.expected}
实际活体：${live.pass ? '通过（需要改进）' : '拒绝（符合预期）'}
` +
      `原因：${live.reason}
分数：${fmt(live.score)}
动作通过：${live.action_pass_count}/${live.required_action_count}
随机闪光：${live.flash_challenge_pass ? '通过' : '拒绝'}
颜色相关：${fmt(flash.color_corr)}，亮度相关：${fmt(flash.brightness_corr)}，响应幅度：${flash.response_amplitude}`;
  } catch (err) {
    $('#securityResult').className = 'result-box bad';
    $('#securityResult').textContent = err.message;
  }
});

$('#randomnessBtn')?.addEventListener('click', async () => {
  $('#securityResult').className = 'result-box';
  $('#securityResult').textContent = '正在生成多组随机挑战...';
  try {
    const data = await api('/api/security/challenge-randomness');
    $('#securityResult').className = 'result-box ok';
    $('#securityResult').textContent =
      `动作池：${data.action_count} 种；每次随机抽取 ${data.group_count} 组\n` +
      `组合空间：${data.total_sequences} 种有序挑战；下方展示前 ${data.generated} 组样例\n` +
      `单组限时：${data.group_timeout_seconds} 秒；整次挑战有效期：${data.ttl_seconds} 秒\n` +
      `每组随机闪光：${data.flash_sequence_length || 0} 段；颜色池：${(data.flash_colors || []).map(c => c.name).join('、')}\n\n` +
      `动作池：${data.actions.map(a => a.label).join('、')}\n\n` +
      data.challenges.map(c => `${c.index}. ${c.labels.join(' -> ')}`).join('\n') +
      `\n\n说明：${data.explain}`;
  } catch (err) {
    $('#securityResult').className = 'result-box bad';
    $('#securityResult').textContent = err.message;
  }
});

$('#reloadScorecardBtn')?.addEventListener('click', loadScorecard);
async function loadScorecard() {
  if (!me) return;
  const data = await api('/api/demo/checklist');
  const labels = {students: '学生', face_samples: '样本', attendance: '考勤', attendance_success: '成功考勤', activities: '活动', participants: '参与记录', emotions: '情绪记录'};
  $('#scorecardSummary').innerHTML = Object.entries(data.counts)
    .map(([k,v]) => `<div class="card"><div class="num">${v}</div><div class="label">${labels[k] || k}</div></div>`).join('');
  const diag = data.emotion_diagnostics || {};
  $('#emotionDiagnostics').innerHTML =
    `<div class="notice"><strong>情绪模型诊断：</strong>` +
    `active=${escapeHtml(diag.active_engine || 'unknown')}；` +
    `emotiefflib=${diag.emotiefflib_loaded ? '已加载' : '未加载'}；` +
    `version=${escapeHtml(diag.emotiefflib_version || '-')}；` +
    `model=${escapeHtml(diag.model_name || '-')}/${escapeHtml(diag.model_engine || '-')}` +
    `${diag.model_error ? `；error=${escapeHtml(diag.model_error)}` : ''}` +
    `</div>`;
  $('#scorecardTable').innerHTML =
    `<thead><tr><th>模块</th><th>得分点</th><th>分值</th><th>现场打开</th><th>证据</th><th>状态</th></tr></thead><tbody>` +
    data.items.map(item => {
      const ready = inferReady(item, data.counts);
      return `<tr><td>${escapeHtml(item.module)}</td><td>${escapeHtml(item.point)}</td><td>${item.score}</td><td><code>${escapeHtml(item.route)}</code></td><td>${escapeHtml(item.evidence)}</td><td class="${ready.ok ? 'score-ok' : 'score-warn'}">${ready.text}</td></tr>`;
    }).join('') + `</tbody>`;
}

function inferReady(item, counts) {
  const p = item.point;
  if (p.includes('人脸库') && counts.face_samples === 0) return {ok:false, text:'需上传样本'};
  if (p.includes('实时渲染') && counts.attendance === 0) return {ok:false, text:'需完成一次考勤'};
  if (p.includes('筛选查询') && counts.attendance === 0) return {ok:false, text:'需产生记录'};
  if (p.includes('Excel') && counts.attendance === 0) return {ok:false, text:'可导出空表，建议先考勤'};
  if (p.includes('合照') && counts.activities === 0) return {ok:false, text:'需上传合照'};
  if (p.includes('活动频次') && counts.participants === 0) return {ok:false, text:'需合照匹配'};
  if (p.includes('情绪') && counts.emotions === 0) return {ok:false, text:'需考勤/合照生成'};
  return {ok:true, text:'可展示'};
}

refreshMe().catch(console.error);

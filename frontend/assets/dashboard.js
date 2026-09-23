const icons = {
 grid:'<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
 chart:'<path d="M3 3v18h18M6 15l4-5 4 3 6-8"/>',
 clock:'<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
 list:'<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>',
 download:'<path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5"/>',
 refresh:'<path d="M20 11a8 8 0 1 0-2 6M20 4v7h-7"/>',
 arrow:'<path d="M4 12h15m-6-6 6 6-6 6"/>',
 info:'<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v.01"/>',
 wind:'<path d="M3 8h12a3 3 0 1 0-3-3M3 12h16a3 3 0 1 1-3 3M3 16h6"/>',
 pause:'<path d="M9 5v14M15 5v14"/>',
 play:'<path d="m8 4 12 8-12 8Z"/>',
 leaf:'<path d="M20 3C3 1 1 13 7 17S22 14 20 3ZM4 21 16 8"/>',
 close:'<path d="m6 6 12 12M6 18 18 6"/>',
 check:'<path d="m5 12 4 4L19 6"/>',
 spark:'<path d="m12 2 2.5 7.5L22 12l-7.5 2.5L12 22l-2.5-7.5L2 12l7.5-2.5Z"/>',
 sun:'<circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1 1m12 12 1 1M5 19l1-1M18 6l1-1"/>'
};
const icon = (name) => `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true">${icons[name] || icons.info}</svg>`;
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = (v, digits=2) => typeof v === 'number' && Number.isFinite(v) ? v.toFixed(digits).replace('.', ',') : '—';
const dateFmt = (v, opts={day:'2-digit',month:'short'}) => v ? new Intl.DateTimeFormat('ru-RU',{timeZone:'UTC',...opts}).format(new Date(v)) : '—';
const timeFmt = v => dateFmt(v, {hour:'2-digit',minute:'2-digit'});
const fullDate = v => dateFmt(v,{day:'numeric',month:'long',year:'numeric',hour:'2-digit',minute:'2-digit'});
const seasonOf = v => { const m = v ? new Date(v).getUTCMonth() : 6; return m < 2 || m===11 ? 'winter' : m<5 ? 'spring' : m<8 ? 'summer' : 'autumn'; };
const seasonName = {winter:'Зима',spring:'Весна',summer:'Лето',autumn:'Осень'};
const modes = {
  fixture:{name:'Демо',help:'Готовые синтетические примеры. Можно изучить интерфейс; реальные прогнозы здесь не рассчитываются.'},
  replay:{name:'Проверка на истории',help:'Прогноз из прошлого: используются только доступные тогда данные. Факт открывается по мере его поступления.'},
  submission:{name:'Прогноз для сдачи',help:'Итоговые выпуски для конкурса. Нужны архивная погода и обученная модель; синтетический экспорт запрещён.'}
};

export function mountDashboard(root, data, sendAction) {
  const local = root._ttLocal || {tab:'forecast', paused:window.matchMedia('(prefers-reduced-motion: reduce)').matches, modal:null, busy:false};
  root._ttLocal = local;
  if (root._ttCleanup) root._ttCleanup();
  local.busy = false;
  const selection = data.selection || {}, result = data.result, chart = data.chart;
  const rows = chart?.rows || [], first = rows[0] || {}, meta = data.provenance || {};
  const turbines = data.catalog.turbines.length ? data.catalog.turbines : [{id:'turbine_1',name:'Турбина 01'},{id:'turbine_2',name:'Турбина 02'}];
  const origins = data.catalog.origins || [], originIndex = Math.max(0, origins.findIndex(o=>o.forecast_id===selection.forecast_id));
  const weather = (result?.weather || []).find(w=>w.turbine_id===selection.turbine_id && Date.parse(w.valid_time)===Date.parse(first.target_start));
  const season = seasonOf(selection.origin_time), isDemo = data.mode==='fixture';
  const selectedName = turbines.find(t=>t.id===selection.turbine_id)?.name || 'Турбина';
  const next = origins[originIndex+1];
  const readiness = data.readiness || {items:[],can_calculate:false};
  const button = (text, action, style='outline', disabled=false) => `<button class="tt-button ${style}" data-action="${action}" ${action==='export'?'title="Готовый CSV полного выпуска для обеих турбин"':''} ${disabled?'disabled':''}>${text}</button>`;
  const powerFor = id => (result?.predictions || []).find(r=>r.turbine_id===id)?.prediction_norm;
  const empty = text => `<div class="tt-empty">${text}</div>`;
  const error = data.error ? `<section class="tt-error" role="alert"><h2>${esc(data.error.title)}</h2><p>${esc(data.error.message)}</p><div class="tt-actions"><code>${esc(data.error.code)}</code>${button('Повторить запрос','refresh')}</div></section>` : '';
  const eventMarkup = data.events.length ? data.events.map(e=>`<div class="tt-agent"><time>${esc(timeFmt(e.event_time))}</time><div><b>${esc(e.agent)}</b><p>${esc(e.action)}</p><small class="tt-small">${esc(e.reason_code)}</small></div><span class="${e.status==='ok'?'ok':'warning'}">${icon(e.status==='ok'?'check':'info')}</span></div>`).join('') : empty('Событий на выбранный момент нет.');

  root.innerHTML = `<div class="tt-app">
    <aside class="tt-sidebar"><div class="tt-symbol" aria-label="TwinTurbo">TT</div><nav class="tt-nav" aria-label="Разделы"><button class="active" data-action="overview" title="Обзор" aria-label="Обзор">${icon('grid')}</button><button data-action="nav-forecast" title="Прогноз" aria-label="Прогноз">${icon('chart')}</button><button data-action="nav-compare" title="Сравнение выпусков" aria-label="Сравнение выпусков">${icon('clock')}</button><button data-action="nav-events" title="Журнал агентов" aria-label="Журнал агентов">${icon('list')}</button></nav><div class="tt-sidebar-footer">CHACHMECHANICS · HACKALEM</div></aside>
    <main class="tt-main">
      <header class="tt-topbar"><div class="tt-top-left"><div class="tt-brand">TwinTurbo<span>.ai</span></div><span class="tt-crumb">Энергетика / Ветровая площадка</span></div><div class="tt-top-right"><span class="tt-pill"><i class="tt-dot"></i>${isDemo?'Синтетические данные':'Архивные выпуски'}</span><select id="mode-select" class="tt-select" aria-label="Режим данных" aria-describedby="mode-help">${Object.entries(modes).map(([key,mode])=>`<option value="${key}" ${data.mode===key?'selected':''}>${mode.name}</option>`).join('')}</select></div></header>
      <p class="tt-mode-help" id="mode-help">${modes[data.mode].help}</p>
      <section class="tt-intro"><div><div class="tt-eyebrow">WIND INTELLIGENCE / 01</div><h1>Энергия ветра. Под контролем.</h1><p>Две турбины. Почасовой прогноз. Каждое решение на виду.</p></div><div class="tt-actions">${button('Следующий выпуск','update','outline',!next || !result)}${button('Рассчитать прогноз','calculate','outline',isDemo || !readiness.can_calculate)}${button(icon('download')+ (isDemo?'Скачать демо CSV':'Скачать CSV'),'export','primary',!result)}</div></section>
      <section class="tt-readiness" aria-label="Подключение данных">${readiness.items.map((item,i)=>`<button class="tt-readiness-item ${esc(item.state)}" data-readiness="${i}" title="${esc(item.detail)}"><i class="tt-dot"></i><span>${esc(item.label)}<strong>${esc(item.value)}</strong></span>${icon('info')}</button>`).join('')}</section>
      ${error}
      <div class="tt-grid">
        <section class="tt-scene" aria-label="Интерактивная схема ветровой площадки"><canvas class="tt-canvas" aria-hidden="true"></canvas>
          <div class="tt-scene-top"><div class="tt-weather"><div class="tt-weather-top"><span class="tt-weather-temp">${num(weather?.temperature_c,0)}°</span><span class="tt-weather-icon">${icon('sun')}</span></div><div class="tt-weather-sub">${esc(weather?.condition || (weather ? 'Облачность не оценена' : 'Нет погодных значений'))}</div><div class="tt-weather-values"><span>${icon('wind')} ${num(weather?.wind_ms,1)} м/с</span><span>${esc(weather?.direction || '—')}</span></div><div class="tt-weather-sub">Прогноз на ${timeFmt(first.target_start)} UTC</div></div><div class="tt-scene-meta"><span>${seasonName[season]} · ${dateFmt(selection.origin_time,{year:'numeric'})}</span><small>${isDemo?'ДЕМОНСТРАЦИОННАЯ ПЛОЩАДКА':'УСЛОВНАЯ СХЕМА ПЛОЩАДКИ'}</small><small>2 ветротурбины</small></div></div>
          ${turbines.slice(0,2).map((t,i)=>`<button class="tt-turbine-hit ${i?'second':''}" data-turbine="${esc(t.id)}" aria-label="Выбрать ${esc(t.name)}" aria-pressed="${selection.turbine_id===t.id}"><span class="tt-label"><span class="tt-label-top"><i class="tt-dot"></i>${esc(t.name)}</span><strong>${num(powerFor(t.id))}<small class="tt-small"> норм.</small></strong></span></button>`).join('')}
          <button class="tt-pause" data-action="pause" title="${local.paused?'Включить анимацию':'Остановить анимацию'}" aria-label="${local.paused?'Включить анимацию':'Остановить анимацию'}">${icon(local.paused?'play':'pause')}</button>
          <div class="tt-scene-caption">${icon('leaf')} Сезон по дате · движение турбин условное</div>
          <div class="tt-timeline"><div class="tt-timeline-head">${button(icon('clock')+' Календарь','calendar','outline',!origins.length)}<strong id="timeline-time">${fullDate(selection.origin_time)} UTC</strong></div><input id="release-timeline" aria-label="Шкала выпусков" type="range" min="0" max="${Math.max(0,origins.length-1)}" value="${originIndex}" step="1" ${origins.length<2?'disabled':''}><div class="tt-dates">${origins.filter((_,i)=>i===0 || i===origins.length-1 || i%Math.max(1,Math.ceil(origins.length/4))===0).map(o=>`<span class="${o.origin_time.slice(0,10)===selection.origin_time?.slice(0,10)?'selected':''}">${dateFmt(o.origin_time)}</span>`).join('')}</div></div>
        </section>
        <aside class="tt-aside"><section class="tt-card"><div class="tt-card-header"><h2>${esc(selectedName)}</h2><span class="tt-status-tag">${result?(isDemo?'DEMO':'ПРОГНОЗ'):'НЕТ ДАННЫХ'}</span></div><div class="tt-small">Прогноз на ${dateFmt(first.target_start)} · ${timeFmt(first.target_start)} UTC</div><div class="tt-power">${num(first.prediction_norm)}<span>норм. ед.</span></div><div class="tt-meter"><i style="width:${(first.prediction_norm ?? 0)*100}%"></i></div><div class="tt-small">Доля от принятой базы нормализации</div><div class="tt-kpis"><div class="tt-kpi"><span>Интервал Q10–Q90</span><strong>${first.q10 == null?'Недоступен':num(first.q10)+' — '+num(first.q90)}</strong></div><div class="tt-kpi"><span>Горизонт прогноза</span><strong>${selection.horizon_hours || '—'} часов</strong></div></div><div class="tt-note">${isDemo?'Все значения — синтетический пример. Качество модели по ним не оценивается.':'Единицы — нормализованная мощность. Для МВт необходимы подтверждённые номиналы.'}</div></section>
        <section class="tt-card"><div class="tt-card-header"><h2>Происхождение данных</h2><button data-action="provenance" class="tt-button outline" style="padding:5px" aria-label="Подробнее о происхождении">${icon('info')}</button></div><dl class="tt-meta-list"><dt>Погодный run</dt><dd>${esc(meta.run_id || '—')}</dd><dt>Возраст run при выпуске</dt><dd>${esc(meta.weather_age || '—')}</dd><dt>Возраст телеметрии</dt><dd>${esc(meta.telemetry_age || '—')}</dd><dt>Версия модели</dt><dd>${esc(meta.model_id || '—')}</dd><dt>Часов показано</dt><dd>${esc(chart?.coverage || '—')}</dd></dl><div class="tt-divider"></div><div class="tt-actions">${button(icon('clock')+' Выпуск и время','controls','outline',!selection.forecast_id)}</div></section></aside>
      </div>
      <section id="analysis-section"><div class="tt-tabs-row"><nav class="tt-tabs" aria-label="Аналитика"><button data-tab="forecast" class="${local.tab==='forecast'?'active':''}">Прогноз мощности</button><button data-tab="compare" class="${local.tab==='compare'?'active':''}">Сравнение</button><button data-tab="events" class="${local.tab==='events'?'active':''}">Журнал агентов</button></nav><div class="tt-segment" aria-label="Горизонт прогноза">${[24,48].map(h=>`<button data-horizon="${h}" class="${selection.horizon_hours===h?'active':''}" ${!result?'disabled':''}>${h} ч</button>`).join('')}</div></div>
      <div class="tt-card tt-chart-card" id="analysis-body"></div></section>
      <div class="tt-below"><section class="tt-card"><div class="tt-card-header"><h2>${icon('spark')} Объяснение результата</h2><span class="tt-small">ADVISOR</span></div><div class="tt-advice">${data.advisor.length?data.advisor.map(t=>`<p>${esc(t)}</p>`).join(''):empty('Объяснение появится вместе с результатом сервиса.')}</div></section><section class="tt-card"><div class="tt-card-header"><h2>Последние действия</h2><span class="tt-small">${isDemo?'ПРИМЕР ЖУРНАЛА':'AS-OF · UTC'}</span></div>${eventMarkup}</section></div>
      <footer class="tt-footer"><span>TwinTurbo.ai · ChachMechanics · HackAlem AI</span><span>${isDemo?'Демонстрация интерфейса · не реальная выработка':'Времена указаны в UTC · только доступные данные'}</span></footer>
    </main></div>`;

  let sceneStop = startScene(root.querySelector('canvas'), season, local.paused, selection.turbine_id, Boolean(weather));
  const act = async action => {
    if(local.busy) return;
    local.busy = true;
    root.querySelector('.tt-main').classList.add('tt-loading');
    try { await sendAction(action); }
    catch { local.busy=false; root.querySelector('.tt-main')?.classList.remove('tt-loading'); showModal('Не удалось связаться с приложением','<p>Проверьте, что локальный сервер запущен, и повторите действие.</p>'); }
  };
  function renderAnalysis() {
    const target = root.querySelector('#analysis-body');
    root.querySelectorAll('[data-tab]').forEach(b=>b.classList.toggle('active',b.dataset.tab===local.tab));
    if(local.tab==='events') { target.innerHTML=`<div class="tt-card-header"><h2>Хронология действий</h2><span class="tt-small">${isDemo?'Синтетический сценарий':'Журнал сервиса'}</span></div>${eventMarkup}<div class="tt-actions" style="margin-top:15px">${button('Обновить журнал','refresh')}${isDemo?button('Показать отказ погоды','failure'):''}</div>`; bindActions(target); return; }
    if(!chart) { target.innerHTML=empty('Нет доступного выпуска. Выберите деморежим или подключите сервис.'); return; }
    if(local.tab==='compare') {
      if(!result.parent_forecast_id) { target.innerHTML=empty('У этого выпуска нет предыдущей версии. Выберите обновлённый выпуск в календаре или нажмите «Следующий выпуск».'); return; }
      if(!selection.compare) { target.innerHTML=empty('Загрузка сравнения…'); act({type:'select',compare:true}); return; }
      const compared=chart.comparison;
      target.innerHTML=`<div class="tt-chart-top"><div><h2>Что изменилось между выпусками</h2><span class="tt-small">${compared.length} общих часов · ${esc(selectedName)} · норм. ед.</span></div><div class="tt-legend"><span><i></i>Новый выпуск</span><span><i class="old"></i>Предыдущий выпуск</span></div></div>${compared.length?`<div class="tt-chart-wrap"><svg class="tt-chart" preserveAspectRatio="none" role="img" aria-label="Сравнение нового и предыдущего прогноза"></svg><div class="tt-tooltip"></div></div><details class="tt-comparison-details"><summary>Точные значения по часам</summary><div class="tt-table-scroll"><table class="tt-table"><thead><tr><th>Целевой час · UTC</th><th>Предыдущий</th><th>Новый</th><th>Изменение</th></tr></thead><tbody>${compared.map(r=>`<tr><td>${dateFmt(r.target_start)} ${timeFmt(r.target_start)}</td><td>${num(r.before,3)}</td><td>${num(r.after,3)}</td><td>${r.delta>0?'+':''}${num(r.delta,3)}</td></tr>`).join('')}</tbody></table></div></details>`:empty('Совпадающих целевых часов нет.')}<p class="tt-small">Сравнение сохранённых выпусков на одинаковых часах. Причина изменения автоматически не определяется.</p>`;
      if(compared.length) renderChart(target.querySelector('svg'),{rows:compared.map(r=>({...r,prediction_norm:r.after})),actuals:[],comparison:compared,comparisonOnly:true});
      return;
    }
    target.innerHTML=`<div class="tt-chart-top"><div><h2>Почасовой прогноз · ${esc(selectedName)}</h2><span class="tt-small">Нормализованная мощность · ${dateFmt(first.target_start)} — ${dateFmt(rows.at(-1)?.target_start)}</span></div><div class="tt-legend"><span><i></i>Прогноз</span><span><i class="band"></i>Q10–Q90</span><span><i class="fact"></i>Доступный факт</span>${selection.compare?'<span><i class="old"></i>Предыдущий</span>':''}</div></div><div class="tt-chart-wrap"><svg class="tt-chart" viewBox="0 0 1080 250" preserveAspectRatio="none" role="img" aria-label="Почасовой прогноз нормализованной мощности"></svg><div class="tt-tooltip"></div></div><div class="tt-chart-foot"><span>${chart.actuals.length?`Факт доступен для ${chart.actuals.length} часов`:'Факт для целевых часов пока недоступен'} · ${rows.some(r=>r.q10!=null)?'Интервал — готовые квантили модели':'Интервал не оценён'}</span><span>Время · UTC</span></div>`;
    renderChart(target.querySelector('svg'),chart);
  }
  function showModal(title,body) {
    root.querySelector('.tt-modal-backdrop')?.remove();
    const modal=document.createElement('div'); modal.className='tt-modal-backdrop';
    modal.innerHTML=`<section class="tt-modal" role="dialog" aria-modal="true" aria-label="${esc(title)}"><div class="tt-card-header"><h2>${esc(title)}</h2><button class="tt-button outline" data-close aria-label="Закрыть">${icon('close')}</button></div>${body}</section>`;
    root.querySelector('.tt-app').appendChild(modal);
    const previousFocus = root.activeElement || document.activeElement;
    const close=()=>{modal.remove(); previousFocus?.focus();};
    modal.querySelector('[data-close]').onclick=close;
    modal.onclick=e=>{if(e.target===modal)close();};
    modal.onkeydown=e=>{
      if(e.key==='Escape')close();
      if(e.key==='Tab') { const focusables=[...modal.querySelectorAll('button,select,input')].filter(el=>!el.disabled); const current=root.activeElement || document.activeElement; const n=focusables.indexOf(current); if(e.shiftKey && n<=0){e.preventDefault();focusables.at(-1)?.focus();} else if(!e.shiftKey && n===focusables.length-1){e.preventDefault();focusables[0]?.focus();} }
    };
    modal.querySelector('[data-close]').focus();
    return modal;
  }
  function controlsModal() {
    const origin=origins[originIndex];
    const modal=showModal('Выпуск и виртуальное время',`<p>Выберите сохранённый выпуск и момент просмотра факта. Время указано в UTC.</p><label for="origin-select">Сохранённый выпуск · UTC</label><select id="origin-select" class="tt-select">${origins.map(o=>`<option value="${esc(o.forecast_id)}" ${o.forecast_id===selection.forecast_id?'selected':''}>${esc(fullDate(o.origin_time))} · ${esc(o.label)}</option>`).join('')}</select><label for="turbine-select">Турбина</label><select id="turbine-select" class="tt-select">${turbines.map(t=>`<option value="${esc(t.id)}" ${t.id===selection.turbine_id?'selected':''}>${esc(t.name)}</option>`).join('')}</select><label for="clock-select">Показывать факт, доступный к моменту</label><select id="clock-select" class="tt-select">${[selection.origin_time,...(origin?.inspection_times || [])].map(t=>`<option value="${esc(t)}" ${t===selection.as_of?'selected':''}>${esc(fullDate(t))} UTC</option>`).join('')}</select><p>Продвижение времени открывает доступные измерения, не меняя сохранённый прогноз. В демо это синтетический факт.</p><div class="tt-actions">${button('Применить','apply','primary')}</div>`);
    modal.querySelector('[data-action=apply]').onclick=()=>act({type:'select',forecast_id:modal.querySelector('#origin-select').value,turbine_id:modal.querySelector('#turbine-select').value,...(modal.querySelector('#origin-select').value===selection.forecast_id?{as_of:modal.querySelector('#clock-select').value}:{})});
  }
  function calculationModal() {
    if(isDemo || !readiness.can_calculate)return;
    const defaultOrigin = selection.origin_time ? new Date(selection.origin_time).toISOString().slice(0,16) : '';
    const modal=showModal('Рассчитать прогноз',`<p>Новый выпуск для ${data.catalog.turbines.length} турбин в режиме «${modes[data.mode].name}». Сервис проверит доступность погоды и модели на выбранный момент.</p><form><label for="calculate-origin">Момент выпуска · UTC</label><input id="calculate-origin" class="tt-select" type="datetime-local" required step="60" value="${esc(defaultOrigin)}"><label for="calculate-horizon">Горизонт</label><select id="calculate-horizon" class="tt-select"><option value="24" ${selection.horizon_hours===24?'selected':''}>24 часа</option><option value="48" ${selection.horizon_hours!==24?'selected':''}>48 часов</option></select><p>Сохранённые прогнозы останутся доступны. Время вводится в UTC, независимо от часового пояса компьютера.</p><button class="tt-button primary" type="submit">Запустить расчёт</button></form>`);
    modal.querySelector('form').onsubmit=e=>{e.preventDefault();const value=modal.querySelector('#calculate-origin').value;if(value)act({type:'calculate',origin_time:new Date(value+'Z').toISOString(),horizon_hours:Number(modal.querySelector('#calculate-horizon').value)});};
  }
  function calendarModal() {
    if(!origins.length)return;
    const months=[...new Set(origins.map(o=>o.origin_time.slice(0,7)))].sort();
    const modal=showModal('Календарь выпусков',`<p>Выделены только даты с сохранёнными прогнозами. После выбора даты укажите время выпуска. Все даты и часы — UTC.</p><label for="calendar-month">Месяц с данными</label><select id="calendar-month" class="tt-select">${months.map(m=>`<option value="${m}" ${m===selection.origin_time?.slice(0,7)?'selected':''}>${dateFmt(m+'-01T00:00:00Z',{month:'long',year:'numeric'})}</option>`).join('')}</select><div class="tt-calendar" aria-label="Даты с выпусками"></div><div class="tt-calendar-releases" aria-live="polite"></div>`);
    const showReleases=day=>{
      const items=origins.filter(o=>o.origin_time.slice(0,10)===day);
      const target=modal.querySelector('.tt-calendar-releases');
      target.innerHTML=`<p>${dateFmt(day+'T00:00:00Z',{day:'numeric',month:'long',year:'numeric'})}</p><div class="tt-actions">${items.map(o=>`<button class="tt-button outline" data-release="${esc(o.forecast_id)}">${timeFmt(o.origin_time)} · ${esc(o.label)}</button>`).join('')}</div>`;
      target.querySelectorAll('[data-release]').forEach(b=>b.onclick=()=>act({type:'select',forecast_id:b.dataset.release}));
      modal.querySelectorAll('[data-day]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.day===day)));
    };
    const drawCalendar=()=>{
      const month=modal.querySelector('#calendar-month').value;
      const firstDay=new Date(month+'-01T00:00:00Z'),offset=(firstDay.getUTCDay()+6)%7;
      const count=new Date(Date.UTC(firstDay.getUTCFullYear(),firstDay.getUTCMonth()+1,0)).getUTCDate();
      const days=new Set(origins.map(o=>o.origin_time.slice(0,10)));
      modal.querySelector('.tt-calendar').innerHTML=['Пн','Вт','Ср','Чт','Пт','Сб','Вс'].map(d=>`<span>${d}</span>`).join('')+'<span></span>'.repeat(offset)+Array.from({length:count},(_,i)=>{
        const day=month+'-'+String(i+1).padStart(2,'0');return `<button data-day="${day}" aria-label="${dateFmt(day+'T00:00:00Z',{day:'numeric',month:'long',year:'numeric'})}" aria-pressed="false" ${days.has(day)?'':'disabled'}>${i+1}</button>`;
      }).join('');
      modal.querySelectorAll('[data-day]:not(:disabled)').forEach(b=>b.onclick=()=>showReleases(b.dataset.day));
      showReleases(selection.origin_time?.startsWith(month)?selection.origin_time.slice(0,10):origins.find(o=>o.origin_time.startsWith(month)).origin_time.slice(0,10));
    };
    modal.querySelector('#calendar-month').onchange=drawCalendar;
    drawCalendar();
  }
  function bindActions(container) {
    container.querySelectorAll('[data-action]').forEach(b=>b.onclick=()=>{
      const action=b.dataset.action;
      if(action==='overview')root.querySelector('.tt-intro').scrollIntoView({behavior:local.paused?'instant':'smooth'});
      else if(action.startsWith('nav-')) {local.tab=action.slice(4);renderAnalysis();root.querySelector('#analysis-section').scrollIntoView({behavior:local.paused?'instant':'smooth'});}
      else if(action==='pause'){local.paused=!local.paused;sceneStop();sceneStop=startScene(root.querySelector('canvas'),season,local.paused,selection.turbine_id,Boolean(weather));b.innerHTML=icon(local.paused?'play':'pause');b.setAttribute('aria-label',local.paused?'Включить анимацию':'Остановить анимацию');}
      else if(action==='provenance')showModal('Паспорт выпуска',`<p>Исходные идентификаторы и временные метки. Возраст данных указан относительно момента выпуска.</p><pre>${esc(JSON.stringify(meta,null,2))}</pre><p>Прогноз ${esc(result?.forecast_id || 'недоступен')}. ${isDemo?'Источник synthetic; конкурсный экспорт запрещён.':''}</p>`);
      else if(action==='controls')controlsModal();
      else if(action==='calendar')calendarModal();
      else if(action==='calculate')calculationModal();
      else if(action==='update' && next)act({type:'select',forecast_id:next.forecast_id,compare:true});
      else if(action==='failure')act({type:'simulate_failure'});
      else if(['refresh','export'].includes(action))act({type:action});
    });
  }
  bindActions(root);
  root.querySelectorAll('[data-readiness]').forEach(b=>b.onclick=()=>{const item=readiness.items[Number(b.dataset.readiness)];showModal(item.label,`<p><strong>${esc(item.value)}</strong></p><p>${esc(item.detail)}</p>`);});
  root.querySelectorAll('[data-turbine]').forEach(b=>b.onclick=()=>act({type:'select',turbine_id:b.dataset.turbine}));
  root.querySelectorAll('[data-horizon]').forEach(b=>b.onclick=()=>act({type:'select',horizon_hours:Number(b.dataset.horizon)}));
  root.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{local.tab=b.dataset.tab;renderAnalysis();});
  root.querySelector('#mode-select').onchange=e=>act({type:'mode',value:e.target.value});
  root.querySelector('#release-timeline').oninput=e=>{root.querySelector('#timeline-time').textContent=fullDate(origins[Number(e.target.value)].origin_time)+' UTC';};
  root.querySelector('#release-timeline').onchange=e=>act({type:'select',forecast_id:origins[Number(e.target.value)].forecast_id});
  renderAnalysis();
  if(data.download) {
    const bytes=Uint8Array.from(atob(data.download.base64),c=>c.charCodeAt(0));
    const url=URL.createObjectURL(new Blob([bytes],{type:data.download.mime}));
    const modal=showModal('Экспорт готов',`<p>Готовый CSV полного выпуска для обеих турбин. ${isDemo?'Синтетические данные помечены DEMO_ONLY; не используйте их для конкурсной сдачи.':''}</p><p>${esc(data.download.filename)}</p><a class="tt-button primary" style="text-decoration:none" data-download>Сохранить CSV</a>`);
    const link=modal.querySelector('[data-download]');link.href=data.download.url || url;link.download=data.download.filename;
    const cleanup=sceneStop;sceneStop=()=>{cleanup();URL.revokeObjectURL(url);};
  }
  root._ttCleanup=()=>sceneStop();
  return root._ttCleanup;
}

function renderChart(svg, chart) {
  const W=Math.max(300,svg.getBoundingClientRect().width),H=250,L=40,R=12,T=15,B=35, rows=chart.rows;
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  if(!rows.length)return;
  const x=i=>L+i*(W-L-R)/Math.max(1,rows.length-1), y=p=>T+(1-p)*(H-T-B);
  const pathFor=(values)=>{let open=false;return values.map((v,i)=>{if(v==null){open=false;return '';} const segment=`${open?'L':'M'}${x(i).toFixed(2)},${y(v).toFixed(2)}`;open=true;return segment;}).join(' ');};
  const actuals=new Map(chart.actuals.map(r=>[r.target_start,r.power_norm]));
  const old=new Map(chart.comparison.map(r=>[r.target_start,r.before]));
  let markup='';
  for(let i=0;i<=4;i++){const v=i/4;markup+=`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="#34402f" stroke-dasharray="3 5"/><text x="0" y="${y(v)+4}" fill="#829276" font-size="11">${num(v,2)}</text>`;}
  let group=[];
  const drawBand=()=>{if(group.length>1){const points=[...group.map(i=>`${x(i)},${y(rows[i].q90)}`),...group.slice().reverse().map(i=>`${x(i)},${y(rows[i].q10)}`)].join(' ');markup+=`<polygon points="${points}" fill="#b0d475" fill-opacity=".11"/>`;}group=[];};
  rows.forEach((r,i)=>{if(r.q10!=null&&r.q90!=null)group.push(i);else drawBand();});drawBand();
  markup+=`<path d="${pathFor(rows.map(r=>r.prediction_norm))}" stroke="#d6f593" fill="none" stroke-width="2.5" vector-effect="non-scaling-stroke"/>`;
  markup+=`<path d="${pathFor(rows.map(r=>actuals.get(r.target_start)))}" stroke="#8cc9c0" fill="none" stroke-width="2" vector-effect="non-scaling-stroke"/>`;
  if(old.size)markup+=`<path d="${pathFor(rows.map(r=>old.get(r.target_start)))}" stroke="#ae9ecd" fill="none" stroke-width="1.5" stroke-dasharray="5 4" vector-effect="non-scaling-stroke"/>`;
  const ticks = new Set(Array.from({length:W<500?4:8},(_,k)=>Math.round(k*(rows.length-1)/((W<500?4:8)-1))));
  rows.forEach((r,i)=>{if(ticks.has(i))markup+=`<text x="${x(i)}" y="${H-10}" fill="#829276" font-size="10" text-anchor="${i===0?'start':i===rows.length-1?'end':'middle'}">${timeFmt(r.target_start)}</text>`;});
  markup+='<line class="hover-line" y1="10" y2="215" stroke="#cbdca288" style="display:none"/>';
  svg.innerHTML=markup;
  const tooltip=svg.parentElement.querySelector('.tt-tooltip');
  svg.onpointermove=e=>{const rect=svg.getBoundingClientRect();const i=Math.max(0,Math.min(rows.length-1,Math.round(((e.clientX-rect.left)/rect.width*W-L)/(W-L-R)*(rows.length-1))));const r=rows[i];const line=svg.querySelector('.hover-line');line.style.display='';line.setAttribute('x1',x(i));line.setAttribute('x2',x(i));tooltip.style.display='block';tooltip.style.left=Math.max(0,Math.min(e.clientX-rect.left+10,rect.width-190))+'px';tooltip.style.top='20px';tooltip.textContent=`${dateFmt(r.target_start)} · ${timeFmt(r.target_start)} UTC\n`+(chart.comparisonOnly?`Новый: ${num(r.prediction_norm,3)}\nПредыдущий: ${num(r.before,3)}\nИзменение: ${r.delta>0?'+':''}${num(r.delta,3)}`:`Прогноз: ${num(r.prediction_norm,3)}\nQ10–Q90: ${r.q10==null?'недоступен':num(r.q10,3)+' — '+num(r.q90,3)}\nФакт: ${num(actuals.get(r.target_start),3)}`);};
  svg.onpointerleave=()=>{tooltip.style.display='none';svg.querySelector('.hover-line').style.display='none';};
}

function startScene(canvas, season, paused, selected, hasWeather) {
  const ctx=canvas.getContext('2d'); if(!ctx)return ()=>{};
  let stopped=false,raf=0,last=0, w=900,h=560;
  const palette={summer:{top:'#43533a',grass:['#455e39','#5c7548','#718651','#304c32','#829657'],soil:'#393b2a',sky:'#384638'},spring:{top:'#436446',grass:['#48744c','#658b50','#81a468','#426947','#9fba6d'],soil:'#343a2c',sky:'#354c40'},autumn:{top:'#605341',grass:['#877348','#a08a51','#645b36','#ba9956','#514d36'],soil:'#45392d',sky:'#514b3c'},winter:{top:'#b1b9b4',grass:['#b0bdb3','#d3d9cf','#849989','#e2e8df','#8a9f92'],soil:'#4b4a42',sky:'#344b4c'}}[season];
  let seed=427; const rand=()=>{seed=(seed*1664525+1013904223)>>>0;return seed/4294967296;};
  const grass=Array.from({length:3200},()=>({x:rand()*2-1,y:rand()*2-1,len:4+rand()*12,tone:Math.floor(rand()*5),phase:rand()*6})).filter(p=>p.x*p.x+p.y*p.y<.94).sort((a,b)=>a.y-b.y);
  const stars=Array.from({length:45},()=>({x:rand(),y:rand(),r:rand()*1.8+.4}));
  const resize=()=>{const rect=canvas.getBoundingClientRect();w=rect.width;h=rect.height;const dpr=Math.min(devicePixelRatio||1,2);canvas.width=w*dpr;canvas.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);draw(last);};
  function turbine(x,base,height,rotor,angle,active){
    const hub=base-height;
    ctx.save();
    ctx.fillStyle='#050f0940';ctx.beginPath();ctx.ellipse(x+26,base+7,46,8,-.18,0,Math.PI*2);ctx.fill();
    if(active){ctx.strokeStyle='#d5f6a499';ctx.lineWidth=1;ctx.beginPath();ctx.ellipse(x,base+1,26,8,0,0,Math.PI*2);ctx.stroke();ctx.strokeStyle='#d5f6a422';ctx.beginPath();ctx.ellipse(x,base+1,36,11,0,0,Math.PI*2);ctx.stroke();}
    const pole=ctx.createLinearGradient(x-7,0,x+9,0);pole.addColorStop(0,'#667c73');pole.addColorStop(.25,'#d9e2d6');pole.addColorStop(.55,'#f4f6e6');pole.addColorStop(1,'#8d9e91');ctx.fillStyle=pole;
    ctx.beginPath();ctx.moveTo(x-3,hub);ctx.lineTo(x+3,hub);ctx.lineTo(x+8,base);ctx.quadraticCurveTo(x,base+5,x-8,base);ctx.closePath();ctx.fill();
    ctx.fillStyle='#889b8e';ctx.beginPath();ctx.roundRect(x-17,hub-5,22,11,5);ctx.fill();
    ctx.save();ctx.translate(x,hub);ctx.rotate(angle);
    for(let b=0;b<3;b++) {ctx.save();ctx.rotate(b*Math.PI*2/3);const blade=ctx.createLinearGradient(-4,0,7,-rotor);blade.addColorStop(0,'#8e9e93');blade.addColorStop(.3,'#dbe3d5');blade.addColorStop(.7,'#eff1e4');blade.addColorStop(1,'#c4cfc1');ctx.fillStyle=blade;ctx.beginPath();ctx.moveTo(-3,3);ctx.bezierCurveTo(-12,-rotor*.3,-5,-rotor*.72,-1,-rotor);ctx.quadraticCurveTo(3,-rotor-6,3,-rotor+1);ctx.bezierCurveTo(5,-rotor*.64,12,-rotor*.2,5,2);ctx.closePath();ctx.fill();ctx.restore();}
    ctx.restore();const hubGrad=ctx.createRadialGradient(x-2,hub-2,0,x,hub,10);hubGrad.addColorStop(0,'#f3f3df');hubGrad.addColorStop(.5,'#d9e1ce');hubGrad.addColorStop(1,'#718977');ctx.fillStyle=hubGrad;ctx.beginPath();ctx.arc(x,hub,8,0,Math.PI*2);ctx.fill();ctx.restore();
  }
  function draw(time){
    if(stopped)return;
    const t=paused?0:time/1000;
    ctx.clearRect(0,0,w,h);
    const bg=ctx.createRadialGradient(w*.55,h*.43,10,w*.5,h*.5,w*.65);bg.addColorStop(0,palette.sky);bg.addColorStop(.65,'#222e27');bg.addColorStop(1,'#15231e');ctx.fillStyle=bg;ctx.fillRect(0,0,w,h);
    // Topographic contour lines are decorative, never geographical coordinates.
    ctx.save();ctx.strokeStyle='#a8bd7820';ctx.lineWidth=.7;for(let k=0;k<9;k++){ctx.beginPath();for(let i=0;i<=100;i++){const a=i/100*Math.PI*2;const rad=135+k*24;const x=w*.55+Math.cos(a)*rad*1.6;const y=h*.66+Math.sin(a)*rad*.46+Math.sin(a*4)*7; i?ctx.lineTo(x,y):ctx.moveTo(x,y);}ctx.closePath();ctx.stroke();}ctx.restore();
    const cx=w*.51,cy=h*.64,rx=w*.42,ry=h*.155,depth=h*.06;
    const islandPath=(offset=0)=>{ctx.beginPath();for(let i=0;i<=120;i++){const a=i/120*Math.PI*2;const rough=1+.013*Math.sin(a*17)+.008*Math.cos(a*29);const x=cx+Math.cos(a)*rx*rough;const y=cy+Math.sin(a)*ry*rough-Math.cos(a)*h*.045+offset;i?ctx.lineTo(x,y):ctx.moveTo(x,y);}ctx.closePath();};
    ctx.save();ctx.shadowColor='#020a08aa';ctx.shadowBlur=38;ctx.shadowOffsetY=23;islandPath(depth);ctx.fillStyle='#111a12';ctx.fill();ctx.restore();
    for(let i=depth;i>=0;i-=2){islandPath(i);ctx.fillStyle=i<4?palette.top:palette.soil;ctx.fill();}
    ctx.save();islandPath();ctx.clip();const ground=ctx.createLinearGradient(0,cy-ry,0,cy+ry);ground.addColorStop(0,palette.top);ground.addColorStop(1,season==='winter'?'#91a49a':'#263d29');ctx.fillStyle=ground;ctx.fillRect(0,cy-ry-60,w,ry*2+120);
    // A soft maintenance path across the patch of land.
    ctx.strokeStyle=season==='winter'?'#81938a':'#72725a';ctx.lineWidth=15;ctx.globalAlpha=.4;ctx.beginPath();ctx.moveTo(w*.27,h*.8);ctx.bezierCurveTo(w*.38,h*.62,w*.5,h*.66,w*.78,h*.53);ctx.stroke();ctx.globalAlpha=1;
    for(const p of grass){const gx=cx+p.x*rx,gy=cy+p.y*ry-p.x*h*.045;const sway=Math.sin(t*1.15+p.phase+p.x*2)*2;ctx.strokeStyle=palette.grass[p.tone];ctx.lineWidth=p.y>.4?1.1:.7;ctx.beginPath();ctx.moveTo(gx,gy);ctx.quadraticCurveTo(gx+1+sway,gy-p.len*.6,gx+4+sway,gy-p.len);ctx.stroke();}
    ctx.restore();
    const mobile=w<550;
    turbine(w*(mobile?.70:.66),h*.58,h*(mobile?.27:.31),h*(mobile?.09:.115),hasWeather?t*.32+.8:.8,selected==='turbine_2');
    turbine(w*(mobile?.37:.39),h*.71,h*.38,h*(mobile?.125:.15),hasWeather?t*.37:0,selected==='turbine_1');
    // Atmospheric flow; illustrative only, not a simulated wake field.
    if(hasWeather){ctx.save();ctx.lineWidth=.8;for(let k=0;k<7;k++){ctx.strokeStyle=`rgba(213,234,176,${.06+(k%3)*.025})`;ctx.beginPath();for(let i=0;i<=80;i++){let x=i/80*w;let y=h*(.4+k*.037)+Math.sin(i/15-t*.4+k)*12+Math.sin(i/10+k)*5; i?ctx.lineTo(x,y):ctx.moveTo(x,y);}ctx.stroke();}ctx.restore();}
    if(season==='winter'){ctx.fillStyle='#dae8df99';stars.forEach(s=>{ctx.beginPath();ctx.arc((s.x*w+t*8)%w,(s.y*h*.7+t*12)%(h*.75),s.r,0,Math.PI*2);ctx.fill();});}
    ctx.fillStyle='#a2b49277';ctx.font='9px "Segoe UI",sans-serif';ctx.fillText('N',w-30,h*.35);ctx.strokeStyle='#92a78655';ctx.beginPath();ctx.moveTo(w-27,h*.36);ctx.lineTo(w-27,h*.42);ctx.stroke();
  }
  function frame(time){if(stopped)return;if(time-last>42){last=time;draw(time);}raf=requestAnimationFrame(frame);}
  const observer=new ResizeObserver(resize);observer.observe(canvas);resize();if(!paused)raf=requestAnimationFrame(frame);
  return ()=>{stopped=true;cancelAnimationFrame(raf);observer.disconnect();};
}

export default function({parentElement,data,setTriggerValue}) {
  const root=parentElement.querySelector('#twinturbo-app');
  return mountDashboard(root,data,action=>setTriggerValue('action',{...action,nonce:Date.now()}));
}

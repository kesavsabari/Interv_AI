import { useEffect, useRef, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { api, API_BASE_URL, tokenStorage } from '../api';
import { ThemeButton } from './ui';
import './live.css';

type Avatar = 'speaking' | 'listening' | 'thinking' | 'idle';
interface Question { id: number; text: string; topic?: string; subtopic?: string; difficulty?: string; is_follow_up?: boolean; time_limit?: number }
interface Answered { n: number; q: string; a: string }

const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.max(0, s) % 60).padStart(2, '0')}`;

function fallbackWsUrl(sessionId: number): string {
  const token = tokenStorage.getAccessToken() || tokenStorage.getGuestToken() || '';
  let host = 'localhost:8000', proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  try { const u = new URL(API_BASE_URL); host = u.host; if (u.protocol === 'https:') proto = 'wss:'; } catch {}
  return `${proto}//${host}/session/${sessionId}/live?token=${encodeURIComponent(token)}`;
}

export default function Live() {
  const { id } = useParams();
  const sessionId = Number(id);
  const nav = useNavigate();
  const loc = useLocation();

  const [conn, setConn] = useState<'connecting' | 'open' | 'lost'>('connecting');
  const [avatar, setAvatar] = useState<Avatar>('idle');
  const [subtitle, setSubtitle] = useState('Connecting to Ava…');
  const [q, setQ] = useState<Question | null>(null);
  const [qNum, setQNum] = useState(0);
  const [plan, setPlan] = useState<{ topic: string; follow: boolean }[]>([]);
  const [draft, setDraft] = useState('');
  const [interim, setInterim] = useState('');
  const [nudge, setNudge] = useState('');
  const [timeLeft, setTimeLeft] = useState(0);
  const [timing, setTiming] = useState(false);
  const [muted, setMuted] = useState(false);
  const [level, setLevel] = useState(0);
  const [banner, setBanner] = useState<{ t: string; m: string } | null>(null);
  const [warnings, setWarnings] = useState(0);
  const [focused, setFocused] = useState(true);
  const [log, setLog] = useState<Answered[]>([]);
  const [elapsed, setElapsed] = useState(0);
  const [finished, setFinished] = useState(false);
  const [confirmEnd, setConfirmEnd] = useState(false);
  const [camOn, setCamOn] = useState(false);

  const ws = useRef<WebSocket | null>(null);
  const stream = useRef<MediaStream | null>(null);
  const video = useRef<HTMLVideoElement>(null);
  const audioEl = useRef<HTMLAudioElement | null>(null);
  const lastSpeak = useRef<{ text: string; url?: string; lang?: string } | null>(null);
  const recog = useRef<any>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const frameTimer = useRef<any>(null);
  const recording = useRef(false);
  const submitted = useRef(true);
  const finalRef = useRef('');  // the answer box text: speech finals plus anything typed
  const typed = useRef(false);
  const interimRef = useRef('');
  // Brave ships webkitSpeechRecognition but blocks Google's service; there the server transcribes with Whisper.
  const serverStt = useRef(!!(navigator as any).brave);
  const qRef = useRef<Question | null>(null);
  const levelRef = useRef(0);
  const speechInChunk = useRef(false);
  const dbSum = useRef({ sum: 0, n: 0 });
  const lastActivity = useRef(0);
  const mutedRef = useRef(false);
  const chunkId = useRef(0);
  const frameId = useRef(0);
  const ended = useRef(false);
  const t0 = useRef(Date.now());

  const send = (msg: Record<string, any>) => {
    if (ws.current?.readyState === WebSocket.OPEN) ws.current.send(JSON.stringify({ ...msg, timestamp: Date.now() }));
  };
  const flash = (t: string, m: string, ms = 4000) => {
    setBanner({ t, m });
    setTimeout(() => setBanner((b) => (b && b.m === m ? null : b)), ms);
  };

  // ---- Camera + mic (one stream for the whole interview) ----
  const ensureStream = async () => {
    if (stream.current?.active) return stream.current;
    try {
      stream.current = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 15 } },
      });
      if (video.current) { video.current.srcObject = stream.current; video.current.play().catch(() => {}); }
      setCamOn(true);
      stream.current.getAudioTracks().forEach((t) => (t.enabled = !mutedRef.current));
      // Mic meter runs for the whole interview; it also tells the server which audio chunks contain speech.
      const ctx = new AudioContext();
      const an = ctx.createAnalyser();
      an.fftSize = 1024;
      ctx.createMediaStreamSource(stream.current).connect(an);
      const buf = new Float32Array(an.fftSize);
      const tick = () => {
        if (ended.current) { ctx.close(); return; }
        an.getFloatTimeDomainData(buf);
        let sum = 0;
        for (const v of buf) sum += v * v;
        const db = 20 * Math.log10(Math.max(Math.sqrt(sum / buf.length), 1e-5));
        const lv = Math.min(1, Math.max(0, (db + 55) / 40));
        dbSum.current.sum += db;
        dbSum.current.n += 1;
        levelRef.current = lv;
        if (lv > 0.3) speechInChunk.current = true;
        setLevel(lv);
        requestAnimationFrame(tick);
      };
      tick();
    } catch {
      flash('Camera or mic blocked', 'Allow camera and microphone in the address bar, then reload.', 8000);
    }
    return stream.current;
  };

  // ---- Recording one answer ----
  const startAnswer = async () => {
    if (recording.current) return;
    recording.current = true;
    submitted.current = false;
    finalRef.current = '';
    interimRef.current = '';
    typed.current = false;
    setDraft('');
    setInterim('');
    const s = await ensureStream();

    // Browser speech recognition: finals are sent once each, so the server transcript never repeats.
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SR) serverStt.current = true;
    if (SR && !serverStt.current) {
      const r = new SR();
      r.continuous = true;
      r.interimResults = true;
      r.lang = 'en-US';
      r.onresult = (e: any) => {
        let inter = '';
        for (let i = e.resultIndex; i < e.results.length; i++) {
          const txt = e.results[i][0].transcript;
          if (mutedRef.current) continue;
          if (e.results[i].isFinal) {
            const piece = txt.trim();
            if (piece) {
              finalRef.current = (finalRef.current + ' ' + piece).trim();
              send({ type: 'audio_chunk', text: piece, is_speech: true, duration: 0.5 });
            }
          } else inter += txt;
        }
        interimRef.current = inter.trim();
        setDraft(finalRef.current);
        setInterim(interimRef.current);
      };
      r.onerror = (e: any) => {
        // Brave/Chromium can't reach Google's recogniser (network) or block it: use the server's Whisper transcript
        if (['network', 'audio-capture', 'not-allowed', 'service-not-allowed'].includes(e.error)) {
          serverStt.current = true;
          if (recog.current === r) recog.current = null;
          try { r.abort(); } catch {}
        }
      };
      // Chrome stops recognition after a pause; restart only this recognizer while still answering.
      r.onend = () => { if (recording.current && recog.current === r) { try { r.start(); } catch {} } };
      try { r.start(); recog.current = r; } catch {}
    }

    // Raw audio for delivery scoring, flagged with whether the mic heard speech in that slice.
    if (s && typeof MediaRecorder !== 'undefined') {
      try {
        const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : 'audio/webm';
        const rec = new MediaRecorder(new MediaStream(s.getAudioTracks()), { mimeType: mime });
        rec.ondataavailable = async (ev) => {
          if (!ev.data.size) return;
          const isSpeech = speechInChunk.current && !mutedRef.current;
          speechInChunk.current = false;
          const { sum, n } = dbSum.current;
          dbSum.current = { sum: 0, n: 0 };
          if (mutedRef.current) return;  // mic off: send nothing, so silence can't be scored or end the answer
          const bytes = new Uint8Array(await ev.data.arrayBuffer());
          let bin = '';
          for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
          chunkId.current += 1;
          send({ type: 'audio_chunk', chunk_id: chunkId.current, data: btoa(bin), format: 'webm', server_transcript: serverStt.current, level_db: n ? Math.round((sum / n) * 10) / 10 : undefined, duration: 0.5, is_speech: isSpeech });
        };
        rec.start(500);
        recorder.current = rec;
      } catch {}
    }

    // One low-res frame per second for presence scoring.
    const canvas = document.createElement('canvas');
    canvas.width = 320;
    canvas.height = 240;
    const c2 = canvas.getContext('2d');
    clearInterval(frameTimer.current);
    frameTimer.current = setInterval(() => {
      const v = video.current;
      if (!v || !c2 || !v.videoWidth) return;
      c2.drawImage(v, 0, 0, 320, 240);
      frameId.current += 1;
      send({ type: 'video_frame', frame_id: frameId.current, frame: canvas.toDataURL('image/jpeg', 0.55) });
    }, 1000);
  };

  const stopAnswer = () => {
    recording.current = false;
    setTiming(false);
    const r = recog.current;
    recog.current = null;
    try { r?.stop(); } catch {}
    if (recorder.current && recorder.current.state !== 'inactive') { try { recorder.current.stop(); } catch {} }
    recorder.current = null;
    clearInterval(frameTimer.current);
  };

  const answerText = () => [finalRef.current, interimRef.current].filter(Boolean).join(' ').trim();

  const done = () => {
    if (submitted.current) return;
    submitted.current = true;
    const text = answerText();
    stopAnswer();
    send({ type: 'answer_ended', text, input_mode: typed.current ? 'typed' : 'voice' });
    if (qRef.current) {
      const cur = qRef.current;
      setLog((l) => [{ n: qNum || l.length + 1, q: cur.text, a: text || '(no answer)' }, ...l]);
    }
    setAvatar('thinking');
    setSubtitle('');
  };

  // ---- Ava's voice ----
  const speak = (text: string, url?: string, lang?: string, aside = false) => {
    // An aside (nudge while you answer) never interrupts the question or restarts the answer timer.
    if (!aside) {
      lastSpeak.current = { text, url, lang };
      audioEl.current?.pause();
      audioEl.current = null;
      speechSynthesis?.cancel();
      setAvatar('speaking');
      setSubtitle(text);
    }
    let finishedSpeaking = false;
    const end = () => {
      if (finishedSpeaking || aside) return;
      finishedSpeaking = true;
      setAvatar((a) => (a === 'speaking' ? 'listening' : a));
      send({ type: 'client_ready' });
    };
    const browserVoice = () => {
      if (!('speechSynthesis' in window)) { setTimeout(end, 1500); return; }
      const u = new SpeechSynthesisUtterance(text);
      u.lang = lang || 'en-US';
      u.onend = end;
      u.onerror = end;
      speechSynthesis.speak(u);
    };
    if (url) {
      const a = new Audio(url.startsWith('http') ? url : `${API_BASE_URL}${url}`);
      if (!aside) audioEl.current = a;
      let fell = false;
      const fall = () => { if (fell || (!aside && audioEl.current !== a)) return; fell = true; browserVoice(); };
      a.onended = end;
      a.onerror = fall;
      a.play().catch(fall);
    } else browserVoice();
  };

  // ---- Socket ----
  useEffect(() => {
    if (!sessionId) { nav('/dashboard'); return; }
    let closed = false, retry: any = null, attempts = 0, lastFrame = Date.now();

    const open = async () => {
      let url = (loc.state as any)?.wsUrl as string | undefined;
      if (!url) { try { url = (await api.getSession(sessionId)).ws_url; } catch { url = fallbackWsUrl(sessionId); } }
      if (location.protocol === 'https:') url = url.replace(/^ws:/, 'wss:');
      if (closed) return;
      setConn('connecting');
      const sock = new WebSocket(url);
      ws.current = sock;
      sock.onopen = () => {
        setConn('open');
        if (attempts > 0) flash('Reconnected', 'Connection dropped briefly. Your answer was kept.', 3000);
        attempts = 0;
        send({ type: 'heartbeat' });
      };
      sock.onmessage = (ev) => {
        lastFrame = Date.now();
        let m: any;
        try { m = JSON.parse(ev.data); } catch { return; }
        switch (m.type) {
          case 'avatar_speak':
            // While answering, Ava's line is a nudge: shown and spoken softly, nothing else changes.
            speak(m.text, m.audio_url, m.language, recording.current);
            break;
          case 'ack':
            setSubtitle(m.text);
            speak(m.text, m.audio_url, m.language, true);
            break;
          case 'question': {
            const isNew = qRef.current?.id !== m.id || qRef.current?.text !== m.text;
            qRef.current = m;
            setQ(m);
            setNudge('');
            setTimeLeft(m.time_limit || 120);
            if (isNew) {
              setQNum((n) => n + 1);
              setPlan((p) => [...p, { topic: m.topic || 'Question', follow: !!m.is_follow_up }]);
              finalRef.current = '';
              interimRef.current = '';
              typed.current = false;
              setDraft('');
              setInterim('');
            }
            break;
          }
          case 'transcript_partial':
            // Whisper's text so far (browsers without live recognition); never overwrite what was typed
            if (recording.current && !typed.current && m.text) {
              finalRef.current = m.text;
              setDraft(m.text);
            }
            break;
          case 'retry_answer':
            // Ava heard speech but got no words: stop, ask again, and listen afresh when she finishes.
            stopAnswer();
            submitted.current = false;
            speak(m.text, m.audio_url, m.language);
            break;
          case 'timer_started':
            setTimeLeft(m.time_limit || 120);
            setTiming(true);
            setAvatar('listening');
            startAnswer();
            break;
          case 'nudge':
          case 'redirect':
            setNudge(m.text || m.message || '');
            break;
          case 'answer_complete':
            if (!submitted.current) {
              submitted.current = true;
              const text = m.text || answerText();
              stopAnswer();
              if (qRef.current) { const cur = qRef.current; setLog((l) => [{ n: l.length + 1, q: cur.text, a: text || '(no answer)' }, ...l]); }
            }
            setAvatar('thinking');
            break;
          case 'integrity_warning':
            setWarnings((w) => m.warning_count ?? w + 1);
            flash('Integrity', m.message || 'Stay on this window and in view of the camera.', 6000);
            break;
          case 'round_end':
            ended.current = true;
            stopAnswer();
            setAvatar('idle');
            setFinished(true);
            setTimeout(() => nav(`/interview/processing/${sessionId}`), 2500);
            break;
          case 'error':
            flash('Server error', m.detail || 'Something went wrong on the server.');
            break;
        }
      };
      sock.onclose = (ev) => {
        if (closed || ended.current) return;
        if (ev.code === 1008) { setConn('lost'); flash('Not allowed', 'This session belongs to another account or has expired.', 10000); return; }
        setConn('lost');
        attempts += 1;
        retry = setTimeout(open, Math.min(10000, 1000 * 1.6 ** attempts));
      };
    };
    open();
    ensureStream();

    const hb = setInterval(() => {
      send({ type: 'heartbeat' });
      if (Date.now() - lastFrame > 45000 && ws.current?.readyState === WebSocket.OPEN) ws.current.close();
    }, 8000);
    const clock = setInterval(() => setElapsed(Math.floor((Date.now() - t0.current) / 1000)), 1000);

    return () => {
      closed = true;
      ended.current = true;
      clearTimeout(retry);
      clearInterval(hb);
      clearInterval(clock);
      stopAnswer();
      stream.current?.getTracks().forEach((t) => t.stop());
      audioEl.current?.pause();
      speechSynthesis?.cancel();
      ws.current?.close();
    };
  }, [sessionId]);

  // Answer countdown; at zero the answer is submitted.
  useEffect(() => {
    if (!timing) return;
    if (timeLeft <= 0) { done(); return; }
    const t = setTimeout(() => setTimeLeft((s) => s - 1), 1000);
    return () => clearTimeout(t);
  }, [timing, timeLeft]);

  // Integrity signals.
  useEffect(() => {
    const ev = (event_type: string) => send({ type: 'integrity_event', event_type, question_id: qRef.current?.id });
    const vis = () => { if (document.hidden) { ev('tab_switch'); setFocused(false); } };
    const blur = () => { ev('window_blur'); setFocused(false); };
    const copy = () => ev('clipboard_copy');
    const paste = () => ev('clipboard_paste');
    document.addEventListener('visibilitychange', vis);
    addEventListener('blur', blur);
    addEventListener('copy', copy);
    addEventListener('paste', paste);
    return () => {
      document.removeEventListener('visibilitychange', vis);
      removeEventListener('blur', blur);
      removeEventListener('copy', copy);
      removeEventListener('paste', paste);
    };
  }, []);

  const toggleMute = () => {
    const m = !mutedRef.current;
    mutedRef.current = m;
    setMuted(m);
    stream.current?.getAudioTracks().forEach((t) => (t.enabled = !m));
    if (m) flash('Mic off', 'Ava can’t hear you. Type your answer, or press M to turn the mic back on.', 3500);
  };

  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement)?.tagName)) return;
      if (e.key === 'm') toggleMute();
      if (e.key === 'r' && lastSpeak.current) speak(lastSpeak.current.text, lastSpeak.current.url, lastSpeak.current.lang);
    };
    addEventListener('keydown', key);
    return () => removeEventListener('keydown', key);
  }, []);

  const endInterview = async () => {
    if (!confirmEnd) { setConfirmEnd(true); setTimeout(() => setConfirmEnd(false), 4000); return; }
    ended.current = true;
    stopAnswer();
    ws.current?.close();
    setFinished(true);
    try { await api.finalizeSession(sessionId); } catch {}
    nav(`/interview/processing/${sessionId}`);
  };

  const words = (draft + ' ' + interim).trim().split(/\s+/).filter(Boolean).length;
  const answering = avatar === 'listening' && timing;
  const stateLabel = avatar === 'speaking' ? 'Ava is asking' : answering ? 'Your turn' : avatar === 'thinking' ? 'Ava is thinking' : conn === 'open' ? 'Waiting' : 'Connecting';
  const avgWords = log.length ? Math.round(log.reduce((s, x) => s + (x.a === '(no answer)' ? 0 : x.a.split(/\s+/).length), 0) / log.length) : 0;
  const follows = plan.filter((p) => p.follow).length;

  return (
    <div className="live">
      <header>
        <span className="brand">IntervAI</span>
        <span className="where">{q?.topic ? `${q.topic}${q.difficulty ? ` · ${q.difficulty}` : ''}` : 'Live interview'}</span>
        <span className="prog">{plan.map((_, i) => <i key={i} className={i < plan.length - 1 ? 'past' : 'now'} />)}</span>
        {answering ? <span className="rec">Recording</span> : null}
        <button className="link" onClick={() => (document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen?.())}>Full screen</button>
        <ThemeButton />
        <button className={`end ${confirmEnd ? 'sure' : ''}`} onClick={endInterview}>{confirmEnd ? 'Click again to end' : 'End interview'}</button>
      </header>
      {banner ? <div className="banner"><b>{banner.t}</b><span>{banner.m}</span></div> : null}
      {conn === 'lost' && !banner ? <div className="banner"><b>Reconnecting</b><span>Lost the connection to the server. Retrying…</span></div> : null}

      <div className="cols">
        <aside>
          <div className="ava">
            <div className={`face ${avatar}`}><i /><i /><i /><i /></div>
            <div><b>Ava</b><span>{avatar === 'speaking' ? 'Speaking' : avatar === 'listening' ? 'Listening' : avatar === 'thinking' ? 'Thinking' : 'Idle'}</span></div>
          </div>
          <p className="sub">{subtitle}</p>
          <ol className="plan">
            {plan.map((p, i) => (
              <li key={i} className={i < plan.length - 1 ? 'past' : 'now'}><i /><span>{p.topic}<small>{p.follow ? 'Follow-up' : `Question ${i + 1}`}</small></span></li>
            ))}
          </ol>
        </aside>

        <section>
          <div className="state"><b>{stateLabel}</b>{qNum ? <span>Question {qNum}</span> : null}{q?.is_follow_up ? <span className="chip">Follow-up</span> : null}{q?.subtopic ? <span className="chip">{q.subtopic}</span> : null}</div>
          <h1 className="serif">{q?.text || (conn === 'open' ? 'Ava is getting ready…' : 'Connecting…')}</h1>
          <div className="answer">
            <div className="lbl"><span>Your answer · live transcript</span><span>{words} words</span></div>
            <div className="t">
              <textarea
                value={draft}
                disabled={!answering}
                placeholder={answering ? (muted ? 'Type your answer…' : 'Speak, or type your answer…') : ''}
                onChange={(e) => {
                  typed.current = true;
                  finalRef.current = e.target.value;
                  setDraft(e.target.value);
                  if (Date.now() - lastActivity.current > 2000) { lastActivity.current = Date.now(); send({ type: 'answer_activity' }); }
                }}
                onKeyDown={(e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); done(); } }}
              />
              {interim ? <p className="interim">{interim}</p> : null}
              <button className={`mic ${muted ? 'off' : ''}`} onClick={toggleMute} title={muted ? 'Turn mic on (M)' : 'Turn mic off and type (M)'} aria-label={muted ? 'Turn microphone on' : 'Turn microphone off'}>
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"><rect x="9" y="3" width="6" height="11" rx="3" /><path d="M5 11a7 7 0 0 0 14 0M12 18v3" />{muted ? <path d="M4 4l16 16" /> : null}</svg>
              </button>
            </div>
            {nudge ? <p className="nudge">{nudge}</p> : null}
          </div>
          <div className="actions">
            <button className="q" disabled={!lastSpeak.current} onClick={() => lastSpeak.current && speak(lastSpeak.current.text, lastSpeak.current.url, lastSpeak.current.lang)}>Repeat</button>
            <button className="btn done" disabled={!answering} onClick={done}>I'm done</button>
          </div>
          {log.length ? (
            <div className="log">
              <div className="lbl" style={{ marginBottom: 8 }}><span>Earlier in this interview</span></div>
              {log.map((e, i) => <div className="e" key={i}><span>Q{e.n}</span><div><p>{e.q}</p><p>{e.a}</p></div></div>)}
            </div>
          ) : null}
        </section>

        <aside>
          <div className="cam"><video ref={video} autoPlay muted playsInline />{!camOn ? <button onClick={ensureStream}>Turn on camera</button> : null}<span>You</span></div>
          <div className="row2">
            <div><div className="k">Time left</div><div className={`time ${timing && timeLeft <= 15 ? 'low' : ''}`}>{timing ? fmt(timeLeft) : '—'}</div></div>
            <div><div className="k">Mic</div><div className="meter">{Array.from({ length: 14 }, (_, i) => <i key={i} style={{ height: 2 + (muted ? 0 : level * 40 * (0.5 + Math.abs(Math.sin(i * 1.7 + elapsed)) * 0.5)) }} />)}</div></div>
          </div>
          <div className="checks">
            <div><span>Window focused</span><span className={focused ? '' : 'bad'}>{focused ? 'Yes' : 'Left once'}</span></div>
            <div><span>Connection</span><span className={conn === 'open' ? '' : 'bad'}>{conn === 'open' ? 'Good' : conn === 'connecting' ? 'Connecting' : 'Lost'}</span></div>
            <div><span>Warnings</span><span className={warnings ? 'bad' : ''}>{warnings}</span></div>
          </div>
          <div className="sess"><div className="k">This session</div>
            <div className="sg">
              <div>Elapsed<b>{fmt(elapsed)}</b></div>
              <div>Answered<b>{log.length}</b></div>
              <div>Avg length<b>{avgWords} words</b></div>
              <div>Follow-ups<b>{follows}</b></div>
            </div>
          </div>
          <p className="dim" style={{ fontSize: 12, marginTop: 'auto', paddingTop: 16 }}>Shortcuts: M mic on/off · R repeat · Ctrl+Enter submit</p>
        </aside>
      </div>

      <div className={`fin ${finished ? 'on' : ''}`}>
        <div>
          <h2>Interview finished</h2>
          <p className="fsub">{log.length} answers · {fmt(elapsed)}. Your report is being prepared.</p>
        </div>
      </div>
    </div>
  );
}

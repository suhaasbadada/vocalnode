import React, { useState, useRef, useEffect } from 'react';
import { Mic, UploadCloud, Play, Loader2, Settings2 } from 'lucide-react';
import { useAudioStreamer } from './hooks/useAudioStreamer';
import { useAudioRecorder } from './hooks/useAudioRecorder';
import './index.css';

interface VoiceProfile {
  id: string;
  name: string;
}

function App() {
  const [text, setText] = useState('');
  const [voiceId, setVoiceId] = useState<string | null>(null);
  const [voices, setVoices] = useState<VoiceProfile[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  
  // Tuning state
  const [speed, setSpeed] = useState(1.0);
  const [prosody, setProsody] = useState(0.8);
  const [cadence, setCadence] = useState(1.2);
  const [tone, setTone] = useState(0.0);

  const [playbackMode, setPlaybackMode] = useState<'streaming' | 'buffered'>('streaming');

  const fileInputRef = useRef<HTMLInputElement>(null);
  const { streamAudio, isPlaying, ttfb, audioDuration, totalGenTime } = useAudioStreamer();
  
  const uploadFile = async (file: File) => {
    const customName = prompt('Enter a name for this voice fingerprint:', file.name.split('.')[0]);
    if (!customName) return; // User cancelled

    setIsUploading(true);
    const formData = new FormData();
    formData.append('audio', file);
    if (customName.trim()) {
      formData.append('name', customName.trim());
    }

    try {
      const res = await fetch('/voice', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (data.status === 'success') {
        setVoiceId(data.voice_id);
        fetchVoices(); // Refresh the list
      }
    } catch (err) {
      console.error('Failed to upload voice', err);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const { isRecording, recordingTime, startRecording, stopRecording } = useAudioRecorder({
    onComplete: (file) => uploadFile(file)
  });

  const fetchVoices = async () => {
    try {
      const res = await fetch('/voices');
      const data = await res.json();
      if (data.status === 'success') {
        setVoices(data.voices);
      }
    } catch (err) {
      console.error('Failed to fetch voices', err);
    }
  };

  useEffect(() => {
    fetchVoices();
  }, []);

  const handleRecordToggle = async () => {
    if (isRecording) {
      const file = await stopRecording();
      if (file) {
        uploadFile(file);
      }
    } else {
      startRecording();
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    uploadFile(file);
  };

  const handleGenerate = () => {
    if (!text.trim()) return;
    streamAudio(text, voiceId, prosody, cadence, tone, speed);
  };

  const appendTag = (tag: string) => {
    setText(prev => prev + (prev.endsWith(' ') || prev.length === 0 ? '' : ' ') + tag + ' ');
  };

  return (
    <div className="app-container" style={{ height: '100vh', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <header className="header" style={{ flexShrink: 0 }}>
        <h1>VocalNode</h1>
        <p><i style={{ color: 'var(--accent)' }}>(Near)</i> Real-time Voice Synthesis Engine</p>
      </header>

      <div className="main-grid" style={{ flex: 1, overflowY: 'auto' }}>
        {/* Left Column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', height: '100%' }}>
          <div className="glass-panel">
            <h2 className="panel-title"><Mic size={24} className="text-accent" /> Voice Fingerprint</h2>

            <input 
              type="file" 
              accept="audio/wav,audio/mpeg" 
              ref={fileInputRef} 
              style={{ display: 'none' }} 
              onChange={handleFileUpload} 
            />
            
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'stretch' }}>
              <div 
                className={`file-dropzone ${voiceId && !voices.find(v => v.id === voiceId) ? 'success' : ''}`}
                onClick={() => fileInputRef.current?.click()}
                style={{ flex: 1, margin: 0 }}
              >
                {isUploading ? (
                  <Loader2 size={32} className="animate-spin text-accent" />
                ) : (
                  <>
                    <UploadCloud size={32} color="var(--accent)" />
                    <div>
                      <p style={{ fontSize: '1rem', fontWeight: 600 }}>Upload Audio</p>
                      <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>.wav or .mp3</p>
                    </div>
                  </>
                )}
              </div>

              <div 
                className={`file-dropzone ${isRecording ? 'recording' : ''}`}
                onClick={handleRecordToggle}
                style={{ flex: 1, margin: 0, borderColor: isRecording ? 'var(--error)' : 'var(--panel-border)', background: isRecording ? 'rgba(239, 68, 68, 0.1)' : '' }}
              >
                {isRecording ? (
                  <>
                    <Mic size={32} color="var(--error)" className="animate-pulse" />
                    <div>
                      <p style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--error)' }}>Recording... {recordingTime}s</p>
                      <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Click to stop (max 10s)</p>
                    </div>
                  </>
                ) : (
                  <>
                    <Mic size={32} color="var(--accent)" />
                    <div>
                      <p style={{ fontSize: '1rem', fontWeight: 600 }}>Record Audio</p>
                      <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Use microphone</p>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>

          <div className="glass-panel">
            <h2 className="panel-title"><Settings2 size={24} className="text-accent" /> Advanced Tuning</h2>
            <div className="tuning-grid">
              <div className="slider-group">
                <div className="slider-header">
                  <span className="slider-label">Speed (Multiplier)</span>
                  <span className="slider-value">{speed.toFixed(2)}x</span>
                </div>
                <input type="range" min="0.5" max="2.0" step="0.05" value={speed} onChange={e => setSpeed(parseFloat(e.target.value))} />
              </div>
              
              <div className="slider-group">
                <div className="slider-header">
                  <span className="slider-label">Prosody (Variance)</span>
                  <span className="slider-value">{prosody.toFixed(2)}</span>
                </div>
                <input type="range" min="0.1" max="2.0" step="0.05" value={prosody} onChange={e => setProsody(parseFloat(e.target.value))} />
              </div>
              
              <div className="slider-group">
                <div className="slider-header">
                  <span className="slider-label">Cadence (Predictability)</span>
                  <span className="slider-value">{cadence.toFixed(2)}</span>
                </div>
                <input type="range" min="1.0" max="2.0" step="0.05" value={cadence} onChange={e => setCadence(parseFloat(e.target.value))} />
              </div>
              
              <div className="slider-group">
                <div className="slider-header">
                  <span className="slider-label">Tone (Exaggeration)</span>
                  <span className="slider-value">{tone.toFixed(2)}</span>
                </div>
                <input type="range" min="0.0" max="1.0" step="0.05" value={tone} onChange={e => setTone(parseFloat(e.target.value))} />
              </div>
            </div>
          </div>

          <div className="glass-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
            <h2 className="panel-title" style={{ fontSize: '1rem', marginBottom: '0.75rem' }}>Paralinguistic Tags</h2>
            <div className="tags-hint" style={{ margin: 0 }}>
              <span onClick={() => appendTag('[laugh]')}>[laugh]</span> 
              <span onClick={() => appendTag('[chuckle]')}>[chuckle]</span> 
              <span onClick={() => appendTag('[sigh]')}>[sigh]</span> 
              <span onClick={() => appendTag('[gasp]')}>[gasp]</span> 
              <span onClick={() => appendTag('[cough]')}>[cough]</span>
              <span onClick={() => appendTag('[clear throat]')}>[clear throat]</span>
              <span onClick={() => appendTag('[sniff]')}>[sniff]</span>
              <span onClick={() => appendTag('[groan]')}>[groan]</span>
              <span onClick={() => appendTag('[shush]')}>[shush]</span>
            </div>
          </div>
        </div>

        {/* Right Column */}
        <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
          
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <h2 className="panel-title" style={{ margin: 0 }}>
                <Play size={24} className="text-accent" /> Synthesizer Studio
              </h2>
            </div>
            
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
              <div className={`eq-container ${isPlaying ? 'active' : ''}`}>
                <div className="eq-bar"></div>
                <div className="eq-bar"></div>
                <div className="eq-bar"></div>
                <div className="eq-bar"></div>
                <div className="eq-bar"></div>
              </div>
              <div className={`on-air-sign ${isPlaying ? 'active' : ''}`}>
                <div className="on-air-dot"></div> ON AIR
              </div>
            </div>
          </div>
          
          <textarea 
            value={text} 
            onChange={(e) => setText(e.target.value)}
            placeholder="Type your speech here..."
            className="panel"
            style={{ flex: 1, minHeight: '200px', marginBottom: '1.5rem' }}
          />

          <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem', alignItems: 'center' }}>
            <span style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-secondary)' }}>PLAYBACK MODE:</span>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.875rem', cursor: 'pointer' }}>
              <input 
                type="radio" 
                name="playbackMode" 
                value="streaming" 
                checked={playbackMode === 'streaming'} 
                onChange={() => setPlaybackMode('streaming')} 
                style={{ accentColor: 'var(--accent)' }}
              />
              Streaming (Fast Response)
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.875rem', cursor: 'pointer' }}>
              <input 
                type="radio" 
                name="playbackMode" 
                value="buffered" 
                checked={playbackMode === 'buffered'} 
                onChange={() => setPlaybackMode('buffered')} 
                style={{ accentColor: 'var(--accent)' }}
              />
              Buffered (Seamless Audio)
            </label>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', alignItems: 'stretch' }}>
            <select 
              value={voiceId || ''} 
              onChange={(e) => setVoiceId(e.target.value || null)}
              style={{
                flex: 1,
                background: 'linear-gradient(180deg, #f9fafb, #d1d5db)',
                border: '1px solid #9ca3af',
                color: '#1f2937',
                padding: '0.75rem 1rem',
                borderRadius: '6px',
                fontFamily: 'inherit',
                fontSize: '1rem',
                fontWeight: 700,
                textTransform: 'uppercase',
                letterSpacing: '1px',
                outline: 'none',
                cursor: 'pointer',
                boxShadow: 'inset 0 2px 0 white, 0 4px 0 #9ca3af, 0 6px 10px rgba(0,0,0,0.1)',
                margin: 0,
                boxSizing: 'border-box'
              }}
            >
              <option value="">Default Voice</option>
              {voices.map(v => (
                <option key={v.id} value={v.id}>{v.name}</option>
              ))}
            </select>

            <button 
              className="btn btn-primary" 
              onClick={() => streamAudio(playbackMode, text, voiceId, prosody, cadence, tone, speed)}
              disabled={!text.trim() || isPlaying}
              style={{ flex: 1, margin: 0, boxSizing: 'border-box', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}
            >
              {isPlaying ? (
                <>
                  <Loader2 size={20} className="animate-spin" /> Streaming...
                </>
              ) : (
                <>
                  <Play size={20} /> Stream Audio
                </>
              )}
            </button>
          </div>

          <div className="hud-container">
            <div className="hud-box">
              <span className="hud-label">Time to First Byte</span>
              <span className="hud-value">
                {ttfb !== null ? `${ttfb} ms` : '--'}
              </span>
            </div>
            <div className="hud-box">
              <span className="hud-label">Total Gen Time</span>
              <span className="hud-value">
                {totalGenTime !== null ? `${(totalGenTime / 1000).toFixed(2)} s` : '--'}
              </span>
            </div>
            <div className="hud-box">
              <span className="hud-label">Audio Length</span>
              <span className="hud-value">
                {audioDuration !== null ? `${audioDuration.toFixed(2)} s` : '--'}
              </span>
            </div>
            <div className="hud-box">
              <span className="hud-label">Status</span>
              <span className="hud-value" style={{ color: isPlaying ? 'var(--success)' : 'var(--text-secondary)' }}>
                {isPlaying ? 'STREAMING' : 'READY'}
              </span>
            </div>
          </div>
        </div>
      </div>

    </div>
  );
}

export default App;

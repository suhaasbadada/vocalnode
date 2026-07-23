import React, { useState, useRef, useEffect } from 'react';
import { Mic, UploadCloud, Play, Loader2, Settings2 } from 'lucide-react';
import { useAudioStreamer } from './hooks/useAudioStreamer';
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

  const fileInputRef = useRef<HTMLInputElement>(null);
  const { streamAudio, isPlaying, ttfb } = useAudioStreamer();

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

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('audio', file);

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

  const handleGenerate = () => {
    if (!text.trim()) return;
    streamAudio(text, voiceId, prosody, cadence, tone, speed);
  };

  const appendTag = (tag: string) => {
    setText(prev => prev + (prev.endsWith(' ') || prev.length === 0 ? '' : ' ') + tag + ' ');
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>VocalNode</h1>
        <p>Real-time Voice Synthesis Engine</p>
      </header>

      <div className="main-grid">
        {/* Left Column */}
        <div>
          <div className="glass-panel">
            <h2 className="panel-title"><Mic size={24} className="text-accent" /> Voice Fingerprint</h2>
            
            {voices.length > 0 && (
              <div style={{ marginBottom: '1.5rem' }}>
                <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>
                  Select a Saved Voice Profile
                </label>
                <select 
                  value={voiceId || ''} 
                  onChange={(e) => setVoiceId(e.target.value || null)}
                  style={{
                    width: '100%',
                    background: 'rgba(0,0,0,0.2)',
                    border: '1px solid var(--panel-border)',
                    color: 'white',
                    padding: '0.75rem',
                    borderRadius: '8px',
                    fontFamily: 'inherit',
                    fontSize: '1rem',
                    outline: 'none'
                  }}
                >
                  <option value="">Default Voice (No Clone)</option>
                  {voices.map(v => (
                    <option key={v.id} value={v.id}>{v.name}</option>
                  ))}
                </select>
              </div>
            )}

            <div style={{ textAlign: 'center', marginBottom: '1rem', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
              — OR UPLOAD A NEW VOICE —
            </div>

            <input 
              type="file" 
              accept="audio/wav,audio/mpeg" 
              ref={fileInputRef} 
              style={{ display: 'none' }} 
              onChange={handleFileUpload} 
            />
            
            <div 
              className={`file-dropzone ${voiceId && !voices.find(v => v.id === voiceId) ? 'success' : ''}`}
              onClick={() => fileInputRef.current?.click()}
            >
              {isUploading ? (
                <Loader2 size={48} className="animate-spin text-accent" />
              ) : (
                <>
                  <UploadCloud size={48} color="var(--accent)" />
                  <div>
                    <p style={{ fontSize: '1.1rem', fontWeight: 600 }}>Upload Audio Reference</p>
                    <p style={{ color: 'var(--text-secondary)' }}>Click to upload a .wav or .mp3 file for voice cloning</p>
                  </div>
                </>
              )}
            </div>
          </div>

          <div className="glass-panel">
            <h2 className="panel-title"><Settings2 size={24} className="text-accent" /> Advanced Tuning</h2>
            <div className="tuning-grid" style={{ gridTemplateColumns: '1fr', gap: '1.25rem' }}>
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
        </div>

        {/* Right Column */}
        <div className="glass-panel" style={{ margin: 0, height: '100%', display: 'flex', flexDirection: 'column' }}>
          <h2 className="panel-title"><Play size={24} className="text-accent" /> Synthesizer Studio</h2>
          
          <textarea 
            placeholder="Enter text to synthesize... Try adding a paralinguistic tag!"
            value={text}
            onChange={(e) => setText(e.target.value)}
            style={{ flex: 1, minHeight: '200px' }}
          />
          
          <div className="tags-hint">
            Available tags: 
            <span onClick={() => appendTag('[laugh]')}>[laugh]</span> 
            <span onClick={() => appendTag('[chuckle]')}>[chuckle]</span> 
            <span onClick={() => appendTag('[sigh]')}>[sigh]</span> 
            <span onClick={() => appendTag('[gasp]')}>[gasp]</span> 
            <span onClick={() => appendTag('[cough]')}>[cough]</span>
          </div>

          <button 
            className="btn btn-primary" 
            onClick={handleGenerate}
            disabled={!text.trim() || isPlaying}
          >
            {isPlaying ? (
              <><Loader2 size={20} className="animate-spin" /> Synthesizing & Playing...</>
            ) : (
              <><Play size={20} /> Generate & Stream Audio</>
            )}
          </button>

          <div className="hud-container">
            <div className="hud-box">
              <span className="hud-label">Time to First Byte</span>
              <span className="hud-value">
                {ttfb !== null ? `${ttfb} ms` : '--'}
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

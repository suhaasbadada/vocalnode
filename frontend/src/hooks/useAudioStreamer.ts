import { useRef, useState } from 'react';

export const useAudioStreamer = () => {
  const audioContextRef = useRef<AudioContext | null>(null);
  const nextPlayTimeRef = useRef<number>(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [ttfb, setTtfb] = useState<number | null>(null);
  
  const getAudioContext = () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    }
    if (audioContextRef.current.state === 'suspended') {
      audioContextRef.current.resume();
    }
    return audioContextRef.current;
  };

  const playChunk = (int16Array: Int16Array, speed: number, sampleRate = 24000) => {
    const ctx = getAudioContext();
    
    // Convert Int16 to Float32 for Web Audio API
    const float32Array = new Float32Array(int16Array.length);
    for (let i = 0; i < int16Array.length; i++) {
      float32Array[i] = int16Array[i] / 32768.0;
    }

    const audioBuffer = ctx.createBuffer(1, float32Array.length, sampleRate);
    audioBuffer.getChannelData(0).set(float32Array);

    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(ctx.destination);

    // Schedule playback continuously
    if (nextPlayTimeRef.current < ctx.currentTime) {
      nextPlayTimeRef.current = ctx.currentTime + 0.05; // 50ms buffer to prevent underruns
    }
    
    source.start(nextPlayTimeRef.current);
    nextPlayTimeRef.current += audioBuffer.duration;
  };

  const streamAudio = async (
    text: string, 
    voiceId: string | null,
    temperature = 0.8,
    repetitionPenalty = 1.2,
    exaggeration = 0.0,
    speed = 1.0
  ) => {
    setIsPlaying(true);
    setTtfb(null);
    nextPlayTimeRef.current = 0;
    
    const startTime = Date.now();
    let firstByteReceived = false;

    try {
      const response = await fetch('/generate-speech', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          text, 
          voice_id: voiceId,
          temperature,
          repetition_penalty: repetitionPenalty,
          exaggeration,
          speed
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate speech');
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No reader available');

      let leftover = new Uint8Array(0);

      while (true) {
        const { done, value } = await reader.read();
        
        if (value && value.length > 0) {
          if (!firstByteReceived) {
            firstByteReceived = true;
            setTtfb(Date.now() - startTime);
          }
          
          // Combine leftover from previous chunk with new value
          const combined = new Uint8Array(leftover.length + value.length);
          combined.set(leftover);
          combined.set(value, leftover.length);

          const safeLength = combined.length - (combined.length % 2);
          leftover = combined.slice(safeLength);

          if (safeLength > 0) {
            // combined is a new ArrayBuffer so byteOffset is guaranteed to be 0
            const int16Array = new Int16Array(combined.buffer, 0, safeLength / 2);
            playChunk(int16Array, 1.0); // We pass 1.0 because speed is now handled by backend
          }
        }

        if (done) break;
      }
    } catch (err) {
      console.error('Streaming error:', err);
    } finally {
      const ctx = getAudioContext();
      const delay = Math.max(0, nextPlayTimeRef.current - ctx.currentTime) * 1000;
      setTimeout(() => setIsPlaying(false), delay + 100);
    }
  };

  return { streamAudio, isPlaying, ttfb };
};

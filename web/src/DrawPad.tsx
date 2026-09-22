import React, { useEffect, useRef } from "react";
import { RotateCcw } from "lucide-react";

export function DrawPad({ disabled, onChange, onStrokeStart }: {
  disabled: boolean;
  onChange: (image: string | null) => void;
  onStrokeStart: () => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const pointer = useRef<number | null>(null);
  const clear = () => {
    const ctx = canvas.current?.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, 512, 512);
  };
  useEffect(clear, []);
  const point = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    return [(event.clientX - rect.left) * 512 / rect.width,
      (event.clientY - rect.top) * 512 / rect.height];
  };
  const finish = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (pointer.current !== event.pointerId) return;
    pointer.current = null;
    onChange(event.currentTarget.toDataURL("image/png"));
  };
  return <div className="draw-pad">
    <div className="draw-heading"><span>Draw with your mouse or finger</span>
      <button className="text-button" disabled={disabled} onClick={() => { clear(); onChange(null); }}>
        <RotateCcw size={14} /> Clear drawing
      </button>
    </div>
    <canvas ref={canvas} width={512} height={512} aria-label="Draw a doodle"
      aria-describedby="drawing-help"
      onPointerDown={event => {
        if (disabled || pointer.current !== null || event.button !== 0) return;
        const ctx = event.currentTarget.getContext("2d");
        if (!ctx) return;
        event.currentTarget.setPointerCapture(event.pointerId);
        pointer.current = event.pointerId;
        onStrokeStart();
        const [x, y] = point(event);
        ctx.fillStyle = "#17291f";
        ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = "#17291f"; ctx.lineWidth = 8;
        ctx.lineCap = "round"; ctx.lineJoin = "round";
        ctx.beginPath(); ctx.moveTo(x, y);
      }}
      onPointerMove={event => {
        if (disabled || pointer.current !== event.pointerId) return;
        const [x, y] = point(event);
        const ctx = event.currentTarget.getContext("2d");
        ctx?.lineTo(x, y); ctx?.stroke();
      }}
      onPointerUp={finish} onPointerCancel={finish} onLostPointerCapture={finish}
    />
    <p id="drawing-help">Draw one of the categories above. Finish your sketch, then hit “Guess doodle” at the top.</p>
  </div>;
}

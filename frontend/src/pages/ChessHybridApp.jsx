import React, { useEffect, useMemo, useRef, useState } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";
import {
  ArrowUpDown,
  Check,
  CircleEqual,
  Copy,
  Cpu,
  Globe,
  Loader2,
  LogIn,
  RotateCcw,
  Share2,
  ShieldAlert,
  Trophy,
  Undo2,
  User,
  Users,
  Volume2,
  VolumeX,
  X,
} from "lucide-react";
import { initSound, playMoveSoundFor, setSoundEnabled } from "@/utils/sound";

const DEFAULT_API_BASE_URL = import.meta.env.PROD
  ? "https://chessengine-2.onrender.com"
  : "http://localhost:8000";
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL).replace(/\/$/, "");
const ENGINE_NAME = "ChessNet";
const LEVELS = ["1", "2", "3", "4", "5", "6"];
const SOUND_KEY = "chessnet.sound";

function buildWebSocketUrl(roomId) {
  if (!roomId) return "";

  if (import.meta.env.VITE_WS_BASE_URL) {
    return `${String(import.meta.env.VITE_WS_BASE_URL).replace(/\/$/, "")}/ws/${roomId}`;
  }

  try {
    const api = new URL(API_BASE_URL);
    const wsProtocol = api.protocol === "https:" ? "wss:" : "ws:";
    return `${wsProtocol}//${api.host}/ws/${roomId}`;
  } catch {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${protocol}//${window.location.host}/ws/${roomId}`;
  }
}

function normalizeFen(fen) {
  return String(fen || "").trim().split(/\s+/).join(" ");
}

function readStoredSound() {
  try {
    return window.localStorage.getItem(SOUND_KEY) !== "off";
  } catch {
    return true;
  }
}

function roomLink(roomId) {
  return `${window.location.origin}${window.location.pathname}?room=${encodeURIComponent(roomId)}`;
}

function clearRoomFromUrl() {
  const url = new URL(window.location.href);
  if (!url.searchParams.has("room")) return;
  url.searchParams.delete("room");
  window.history.replaceState(null, "", url.toString());
}

// Text presentation selector: without it iOS draws the pawn as an emoji.
const PIECE_GLYPHS = { p: "♟︎", n: "♞︎", b: "♝︎", r: "♜︎", q: "♛︎" };
const PIECE_POINTS = { p: 1, n: 3, b: 3, r: 5, q: 9 };
const START_COUNTS = { p: 8, n: 2, b: 2, r: 2, q: 1 };

// Pieces each side has taken (by the colour that took them) and the material balance.
function materialInfo(game) {
  const onBoard = { w: { p: 0, n: 0, b: 0, r: 0, q: 0 }, b: { p: 0, n: 0, b: 0, r: 0, q: 0 } };
  let balance = 0;
  for (const row of game.board()) {
    for (const piece of row) {
      if (!piece || piece.type === "k") continue;
      onBoard[piece.color][piece.type] += 1;
      balance += (piece.color === "w" ? 1 : -1) * PIECE_POINTS[piece.type];
    }
  }
  const takenBy = { w: [], b: [] };
  for (const type of ["q", "r", "b", "n", "p"]) {
    for (let i = onBoard.b[type]; i < START_COUNTS[type]; i += 1) takenBy.w.push(type);
    for (let i = onBoard.w[type]; i < START_COUNTS[type]; i += 1) takenBy.b.push(type);
  }
  return { takenBy, balance };
}

function isEnPassantMove(move) {
  return typeof move?.flags === "string" && move.flags.includes("e");
}

function enPassantCapturedSquare(move) {
  if (!isEnPassantMove(move)) return null;
  return `${move.to[0]}${move.from[1]}`;
}

function formatHistoryMove(move) {
  if (!isEnPassantMove(move)) return move.san;
  return move.san.includes("e.p.") ? move.san : `${move.san} e.p.`;
}

function findKingSquare(game, color) {
  const board = game.board();
  for (let rowIndex = 0; rowIndex < board.length; rowIndex += 1) {
    for (let colIndex = 0; colIndex < board[rowIndex].length; colIndex += 1) {
      const piece = board[rowIndex][colIndex];
      if (piece?.type === "k" && piece.color === color) {
        return `${String.fromCharCode(97 + colIndex)}${8 - rowIndex}`;
      }
    }
  }
  return null;
}

function getDrawReason(game) {
  if (game.isStalemate?.()) return "Stalemate: the side to move has no legal move and is not in check.";
  if (game.isInsufficientMaterial?.()) return "Insufficient material: neither side can force checkmate.";
  if (game.isThreefoldRepetition?.()) return "Threefold repetition: the same position occurred three times.";
  if (game.isDrawByFiftyMoves?.()) return "Fifty-move rule: 50 moves without a pawn move or capture.";
  return "The game ended in a draw.";
}

function getGameEndInfo(game, playerColor, isMultiplayer) {
  if (!game.isGameOver()) return null;

  if (game.isCheckmate()) {
    const winnerColor = game.turn() === "w" ? "b" : "w";
    const winnerName = winnerColor === "w" ? "White" : "Black";
    const playerWon = winnerColor === playerColor;
    return {
      tone: playerWon ? "win" : "loss",
      title: isMultiplayer ? `${winnerName} wins` : playerWon ? "You won" : "You lost",
      reason: `Checkmate. ${winnerName} trapped the king.`,
      icon: playerWon ? "trophy" : "alert",
    };
  }

  return { tone: "draw", title: "Draw", reason: getDrawReason(game), icon: "draw" };
}

// Game history for the engine: with it the server sees repetitions and keeps its
// search tree from one move to the next (moves are UCI from the history's start).
function historyPayload(game) {
  const history = game.history({ verbose: true });
  if (history.length === 0) return {};
  return { start_fen: history[0].before, moves: history.map((move) => move.lan) };
}

function createGameId() {
  return globalThis.crypto?.randomUUID?.() || `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function EndIcon({ icon, className }) {
  if (icon === "draw") return <CircleEqual className={className} />;
  if (icon === "trophy") return <Trophy className={className} />;
  return <ShieldAlert className={className} />;
}

function ThinkingDots() {
  return (
    <span className="inline-flex items-center gap-0.5" aria-label="thinking">
      {[0, 150, 300].map((delay) => (
        <span key={delay} className="h-1.5 w-1.5 animate-bounce rounded-full bg-emerald-300" style={{ animationDelay: `${delay}ms` }} />
      ))}
    </span>
  );
}

function PlayerStrip({ name, detail, color, isEngine, active, thinking, taken, advantage }) {
  const takenColor = color === "w" ? "text-zinc-500" : "text-zinc-100";
  return (
    <div
      className={`flex h-12 items-center gap-2.5 rounded-xl px-2 transition-colors ${
        active ? "bg-zinc-800/90 ring-1 ring-emerald-400/40" : "bg-zinc-900/70"
      }`}
    >
      <div
        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${
          color === "w" ? "bg-zinc-100 text-zinc-900" : "bg-zinc-950 text-zinc-100 ring-1 ring-zinc-700"
        }`}
      >
        {isEngine ? <Cpu className="h-4 w-4" /> : <User className="h-4 w-4" />}
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2 text-sm leading-tight">
          <span className="truncate font-semibold text-zinc-100">{name}</span>
          {detail && <span className="shrink-0 text-xs text-zinc-500">{detail}</span>}
          {thinking && <ThinkingDots />}
        </div>
        <div className="flex h-4 items-center overflow-hidden text-[13px] leading-none">
          <span className={`truncate tracking-tighter ${takenColor}`}>{taken.map((type) => PIECE_GLYPHS[type]).join("")}</span>
          {advantage > 0 && <span className="ml-1 shrink-0 text-xs font-medium text-zinc-400">+{advantage}</span>}
        </div>
      </div>
      {active && !thinking && <span className="shrink-0 rounded-md bg-emerald-400/15 px-1.5 py-0.5 text-[11px] font-medium text-emerald-300">to move</span>}
    </div>
  );
}

function Segmented({ options, value, onChange, disabled, label }) {
  return (
    <div role="radiogroup" aria-label={label} className="grid gap-1 rounded-xl bg-zinc-950 p-1 ring-1 ring-zinc-800" style={{ gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` }}>
      {options.map((option) => {
        const selected = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={selected}
            disabled={disabled}
            onClick={() => onChange(option.value)}
            className={`h-9 rounded-lg text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
              selected ? "bg-zinc-100 text-zinc-900" : "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100"
            }`}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

function Panel({ title, icon, children, className = "" }) {
  return (
    <section className={`rounded-2xl border border-zinc-800/80 bg-zinc-900/80 p-4 ${className}`}>
      {title && (
        <h2 className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">
          {icon}
          {title}
        </h2>
      )}
      {children}
    </section>
  );
}

function ActionButton({ children, className = "", ...rest }) {
  return (
    <button
      type="button"
      className={`flex h-11 items-center justify-center gap-2 rounded-xl text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${className}`}
      {...rest}
    >
      {children}
    </button>
  );
}

export default function ChessHybridApp() {
  const gameRef = useRef(new Chess());
  const boardAreaRef = useRef(null);
  const desktopMovesRef = useRef(null);
  const mobileMovesRef = useRef(null);
  const wsRef = useRef(null);
  const engineWarmupPromiseRef = useRef(null);
  const engineRequestRef = useRef(0);
  const gameIdRef = useRef(null);
  const confirmTimerRef = useRef(null);

  const [fen, setFen] = useState(gameRef.current.fen());
  const [boardWidth, setBoardWidth] = useState(360);
  const [moveHistory, setMoveHistory] = useState([]);
  const [playerColor, setPlayerColor] = useState("w");
  const [nextColor, setNextColor] = useState("w");
  const [flipped, setFlipped] = useState(false);
  const [depth, setDepth] = useState("6");
  const [engineThinking, setEngineThinking] = useState(false);
  const [engineWarmupStatus, setEngineWarmupStatus] = useState("idle");
  const [warmupStartedAt, setWarmupStartedAt] = useState(0);
  const [now, setNow] = useState(Date.now());
  const [lastMoveSquares, setLastMoveSquares] = useState({});
  const [notice, setNotice] = useState(null);
  const [moveFrom, setMoveFrom] = useState("");
  const [optionSquares, setOptionSquares] = useState({});
  const [endDismissed, setEndDismissed] = useState(false);
  const [confirmNewGame, setConfirmNewGame] = useState(false);
  const [soundOn, setSoundOn] = useState(readStoredSound);

  const [isMultiplayer, setIsMultiplayer] = useState(false);
  const [roomId, setRoomId] = useState("");
  const [roomStatus, setRoomStatus] = useState("idle");
  const [joinOpen, setJoinOpen] = useState(false);
  const [joinCode, setJoinCode] = useState("");
  const [copied, setCopied] = useState(false);

  const [promotionMoveDetail, setPromotionMoveDetail] = useState(null);

  const game = gameRef.current;
  const playerTurn = game.turn() === playerColor;
  const gameOver = game.isGameOver();
  const engineWaking = !isMultiplayer && engineWarmupStatus === "waking";
  const warmupSeconds = warmupStartedAt ? Math.max(0, Math.round((now - warmupStartedAt) / 1000)) : 0;
  const canMove = !gameOver && playerTurn && (isMultiplayer ? roomStatus === "open" : !engineThinking);
  const playerHasMoved = moveHistory.length > (playerColor === "b" ? 1 : 0);

  const boardOrientation = (playerColor === "w") !== flipped ? "white" : "black";
  const bottomColor = boardOrientation === "white" ? "w" : "b";
  const topColor = bottomColor === "w" ? "b" : "w";

  const checkedKingSquare = useMemo(() => (game.isCheck() ? findKingSquare(game, game.turn()) : null), [fen, game]);
  const gameEndInfo = useMemo(() => getGameEndInfo(game, playerColor, isMultiplayer), [fen, playerColor, isMultiplayer, game]);
  const material = useMemo(() => materialInfo(game), [fen, game]);

  useEffect(() => {
    setSoundEnabled(soundOn);
    try {
      window.localStorage.setItem(SOUND_KEY, soundOn ? "on" : "off");
    } catch {
      // storage can be unavailable (private mode); the toggle still works for this visit
    }
  }, [soundOn]);

  useEffect(() => {
    initSound();
    const code = new URLSearchParams(window.location.search).get("room");
    if (code) joinRoom(code);
    else void warmupEngineServer();
    return () => window.clearTimeout(confirmTimerRef.current);
  }, []);

  useEffect(() => {
    if (engineWarmupStatus !== "waking") return undefined;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [engineWarmupStatus]);

  // The board fills the column on phones; on laptops it is also capped by the window
  // height so the whole board and both player strips fit without scrolling.
  useEffect(() => {
    const area = boardAreaRef.current;
    if (!area) return undefined;

    const update = () => {
      const width = Math.floor(area.getBoundingClientRect().width);
      if (width <= 0) return;
      const desktop = window.matchMedia("(min-width: 1024px)").matches;
      const byHeight = window.innerHeight - (desktop ? 190 : 150);
      const cap = Math.min(720, desktop ? byHeight : Math.max(byHeight, 320));
      setBoardWidth(Math.min(width, Math.max(240, cap)));
    };
    const observer = new ResizeObserver(update);
    observer.observe(area);
    window.addEventListener("resize", update);
    update();
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", update);
    };
  }, []);

  useEffect(() => {
    const desktop = desktopMovesRef.current;
    if (desktop) desktop.scrollTop = desktop.scrollHeight;
    const mobile = mobileMovesRef.current;
    if (mobile) mobile.scrollLeft = mobile.scrollWidth;
  }, [moveHistory]);

  useEffect(() => {
    if (!isMultiplayer || !roomId) return undefined;

    setRoomStatus("connecting");
    const socket = new WebSocket(buildWebSocketUrl(roomId));
    wsRef.current = socket;

    socket.onopen = () => setRoomStatus("open");
    socket.onclose = () => setRoomStatus((status) => (status === "left" ? status : "closed"));
    socket.onerror = () => setRoomStatus("closed");
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "move") {
        applyRoomMove(data.move, data.fen);
      } else if ((data.type === "init" || data.type === "error") && data.fen) {
        if (normalizeFen(data.fen) !== normalizeFen(game.fen())) loadAuthoritativeFen(data.fen, data.type);
        if (data.type === "error") setNotice({ tone: "error", text: data.error || "Move rejected by the room" });
      }
    };

    return () => {
      socket.onclose = null;
      socket.close();
      if (wsRef.current === socket) wsRef.current = null;
    };
  }, [isMultiplayer, roomId]);

  const customSquareStyles = useMemo(() => {
    const styles = { ...lastMoveSquares, ...optionSquares };
    if (moveFrom) {
      styles[moveFrom] = { ...styles[moveFrom], background: "rgba(255, 255, 0, 0.4)" };
    }
    if (checkedKingSquare) {
      styles[checkedKingSquare] = {
        ...styles[checkedKingSquare],
        background: "radial-gradient(circle, rgba(248, 113, 113, 0.95) 0%, rgba(220, 38, 38, 0.75) 45%, transparent 75%)",
      };
    }
    return styles;
  }, [checkedKingSquare, lastMoveSquares, moveFrom, optionSquares]);

  function syncGame() {
    setFen(game.fen());
    setMoveHistory(game.history({ verbose: true }).map(formatHistoryMove));
  }

  function clearSelection() {
    setMoveFrom("");
    setOptionSquares({});
  }

  function loadAuthoritativeFen(nextFen, context) {
    if (!nextFen) return false;
    try {
      game.load(nextFen);
      setLastMoveSquares({});
      syncGame();
      return true;
    } catch (err) {
      console.error("Failed to load authoritative FEN", { context, nextFen, err });
      return false;
    }
  }

  function highlightLastMove(move) {
    if (!move) {
      setLastMoveSquares({});
      return;
    }
    const styles = {
      [move.from]: { background: "rgba(250, 204, 21, 0.35)" },
      [move.to]: { background: "rgba(250, 204, 21, 0.35)" },
    };
    const capturedSquare = enPassantCapturedSquare(move);
    if (capturedSquare) {
      styles[capturedSquare] = {
        background: "rgba(239, 68, 68, 0.38)",
        boxShadow: "inset 0 0 0 3px rgba(248, 113, 113, 0.75)",
      };
      setNotice({ tone: "info", text: `${formatHistoryMove(move)}: captured en passant on ${capturedSquare}` });
    }
    setLastMoveSquares(styles);
  }

  function playLocalMove(details) {
    const move = game.move(details);
    clearSelection();
    setNotice(null);
    highlightLastMove(move);
    syncGame();
    playMoveSoundFor(move, game);

    if (isMultiplayer) {
      wsRef.current?.send(JSON.stringify({ type: "move", move: `${move.from}${move.to}${move.promotion || ""}` }));
    } else {
      void maybePlayEngine();
    }
    return move;
  }

  // Room broadcasts include our own moves; replaying the move (instead of loading the
  // FEN) keeps the move list, which a FEN load would wipe.
  function applyRoomMove(uci, roomFen) {
    if (normalizeFen(game.fen()) === normalizeFen(roomFen)) return;
    let move = null;
    try {
      move = uci ? game.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] }) : null;
    } catch {
      move = null;
    }
    if (move && normalizeFen(game.fen()) === normalizeFen(roomFen)) {
      highlightLastMove(move);
      syncGame();
      playMoveSoundFor(move, game);
      return;
    }
    loadAuthoritativeFen(roomFen, "room-move");
  }

  function applyEngineMove(uci, authoritativeFen) {
    const moveDetails = { from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci.length === 5 ? uci[4] : "q" };

    let move;
    try {
      move = game.move(moveDetails);
    } catch (err) {
      if (loadAuthoritativeFen(authoritativeFen, "engine-move-error")) return;
      throw err;
    }

    if (!move) {
      loadAuthoritativeFen(authoritativeFen, "engine-move-null");
      return;
    }

    highlightLastMove(move);
    syncGame();
    playMoveSoundFor(move, game);

    if (authoritativeFen && normalizeFen(game.fen()) !== normalizeFen(authoritativeFen)) {
      console.warn("Frontend/backend FEN mismatch after engine move", { uci, localFen: game.fen(), authoritativeFen });
      loadAuthoritativeFen(authoritativeFen, "engine-move-mismatch");
    }
  }

  async function warmupEngineServer() {
    if (engineWarmupStatus === "ready") return true;
    if (engineWarmupPromiseRef.current) return engineWarmupPromiseRef.current;

    setEngineWarmupStatus("waking");
    setWarmupStartedAt(Date.now());
    setNow(Date.now());
    const warmupPromise = fetch(`${API_BASE_URL}/health`, { cache: "no-store" })
      .then(async (res) => {
        if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
        const data = await res.json().catch(() => ({}));
        if (data.model === false) throw new Error(`${ENGINE_NAME} model is not loaded`);
        setEngineWarmupStatus("ready");
        return true;
      })
      .catch((err) => {
        console.error("Engine warmup failed", err);
        setEngineWarmupStatus("error");
        return false;
      })
      .finally(() => {
        engineWarmupPromiseRef.current = null;
      });

    engineWarmupPromiseRef.current = warmupPromise;
    return warmupPromise;
  }

  async function maybePlayEngine(activePlayerColor = playerColor) {
    if (game.isGameOver()) return;
    if (game.turn() === activePlayerColor) return;

    const requestId = ++engineRequestRef.current;
    const requestFen = game.fen();
    setEngineThinking(true);
    try {
      const serverReady = await warmupEngineServer();
      if (!serverReady) throw new Error("The engine server did not answer");

      const engineDepth = Number(depth);
      if (!gameIdRef.current) gameIdRef.current = createGameId();
      const res = await fetch(`${API_BASE_URL}/fastmove`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          fen: requestFen,
          game_id: gameIdRef.current,
          topk: Math.max(10, Math.min(16, engineDepth * 3)),
          depth: engineDepth,
          adaptive: true,
          ...historyPayload(game),
        }),
      });
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Engine request failed (${res.status})`);
      }
      const data = await res.json();
      if (requestId !== engineRequestRef.current) return;

      const uci = data.move || data.moves?.[0]?.uci || data.best_move;
      if (!uci) throw new Error("The engine returned no move");
      if (normalizeFen(game.fen()) !== normalizeFen(requestFen)) {
        console.warn("Ignoring stale engine response", { uci, requestFen, currentFen: game.fen() });
        return;
      }
      applyEngineMove(uci, data.fen_after);
    } catch (err) {
      if (requestId !== engineRequestRef.current) return;
      console.error("AI request failed", err);
      setNotice({ tone: "error", text: err instanceof Error ? err.message : "Engine move failed", retry: true });
    } finally {
      if (requestId === engineRequestRef.current) setEngineThinking(false);
    }
  }

  function cancelEngineRequest() {
    engineRequestRef.current += 1;
    setEngineThinking(false);
  }

  function getMoveOptions(square) {
    const moves = game.moves({ square, verbose: true });
    const options = {};
    moves.forEach((m) => {
      options[m.to] = {
        background: game.get(m.to)
          ? "radial-gradient(circle, transparent 58%, rgba(0,0,0,.28) 59%, rgba(0,0,0,.28) 78%, transparent 79%)"
          : "radial-gradient(circle, rgba(0,0,0,.28) 24%, transparent 25%)",
      };
    });
    setOptionSquares(options);
    return moves;
  }

  // Dropping or tapping the king on its own rook means castling.
  function castlingTarget(from, to) {
    const source = game.get(from);
    const target = game.get(to);
    if (source?.type !== "k" || target?.type !== "r" || source.color !== target.color) return to;
    return { e1h1: "g1", e1a1: "c1", e8h8: "g8", e8a8: "c8" }[`${from}${to}`] || to;
  }

  function tryMove(from, rawTo) {
    const to = castlingTarget(from, rawTo);
    const legal = game.moves({ square: from, verbose: true }).filter((m) => m.to === to);
    if (!legal.length) return false;
    if (legal.some((m) => m.promotion)) {
      clearSelection();
      setPromotionMoveDetail({ from, to });
      return "promotion";
    }
    playLocalMove({ from, to });
    return true;
  }

  function onDrop(sourceSquare, targetSquare) {
    if (!canMove) return false;
    // A promotion returns false so the pawn snaps back while the piece is chosen.
    return tryMove(sourceSquare, targetSquare) === true;
  }

  function onSquareClick(square) {
    if (!canMove) return;
    if (square === moveFrom) {
      clearSelection();
      return;
    }
    if (moveFrom && tryMove(moveFrom, square)) return;

    const piece = game.get(square);
    if (piece && piece.color === playerColor) {
      setMoveFrom(square);
      getMoveOptions(square);
    } else {
      clearSelection();
    }
  }

  function handlePromotionSelect(pieceType) {
    const detail = promotionMoveDetail;
    setPromotionMoveDetail(null);
    if (!detail) return;
    try {
      playLocalMove({ ...detail, promotion: pieceType });
    } catch {
      clearSelection();
    }
  }

  function resetBoardState() {
    cancelEngineRequest();
    gameIdRef.current = null;
    game.reset();
    syncGame();
    setLastMoveSquares({});
    setNotice(null);
    clearSelection();
    setPromotionMoveDetail(null);
    setEndDismissed(false);
    setConfirmNewGame(false);
  }

  function newGame(choice = nextColor) {
    const color = choice === "random" ? (Math.random() < 0.5 ? "w" : "b") : choice;
    resetBoardState();
    setPlayerColor(color);
    setFlipped(false);
    if (color === "b") void maybePlayEngine(color);
  }

  function requestNewGame() {
    if (isMultiplayer) return;
    if (!playerHasMoved || gameOver || confirmNewGame) {
      newGame();
      return;
    }
    setConfirmNewGame(true);
    window.clearTimeout(confirmTimerRef.current);
    confirmTimerRef.current = window.setTimeout(() => setConfirmNewGame(false), 3000);
  }

  function chooseColor(choice) {
    setNextColor(choice);
    if (!playerHasMoved && !isMultiplayer) newGame(choice);
  }

  function takeBack() {
    if (isMultiplayer || !playerHasMoved) return;
    cancelEngineRequest();
    do {
      game.undo();
    } while (game.turn() !== playerColor && game.history().length > 0);
    const history = game.history({ verbose: true });
    highlightLastMove(history[history.length - 1] || null);
    setNotice(null);
    clearSelection();
    setEndDismissed(false);
    syncGame();
  }

  function hostRoom() {
    const code = Math.random().toString(36).substring(2, 7);
    resetBoardState();
    setPlayerColor("w");
    setFlipped(false);
    setRoomId(code);
    setIsMultiplayer(true);
    setJoinOpen(false);
  }

  function joinRoom(rawCode) {
    const code = String(rawCode || "").trim().toLowerCase();
    if (!code) return;
    resetBoardState();
    setPlayerColor("b");
    setFlipped(false);
    setRoomId(code);
    setIsMultiplayer(true);
    setJoinOpen(false);
    setJoinCode("");
  }

  function leaveRoom() {
    setRoomStatus("left");
    setIsMultiplayer(false);
    setRoomId("");
    clearRoomFromUrl();
    void warmupEngineServer();
    newGame();
  }

  async function shareRoom() {
    const link = roomLink(roomId);
    if (navigator.share) {
      try {
        await navigator.share({ title: "Play chess with me", url: link });
        return;
      } catch {
        // cancelled or unsupported: fall back to copying
      }
    }
    try {
      await navigator.clipboard.writeText(link);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      setNotice({ tone: "info", text: `Room link: ${link}` });
    }
  }

  const status = (() => {
    if (gameEndInfo) return gameEndInfo.title;
    if (isMultiplayer && roomStatus === "connecting") return "Connecting to the room…";
    if (isMultiplayer && roomStatus === "closed") return "Disconnected from the room";
    if (!isMultiplayer && engineThinking) return engineWaking ? `Waking up ${ENGINE_NAME}…` : `${ENGINE_NAME} is thinking…`;
    const check = game.isCheck() ? " · check!" : "";
    if (playerTurn) return `Your move${check}`;
    return `${isMultiplayer ? "Opponent's" : `${ENGINE_NAME}'s`} move${check}`;
  })();

  const endTheme = gameEndInfo?.tone === "win"
    ? "border-emerald-400/40 bg-emerald-500/15 text-emerald-100"
    : gameEndInfo?.tone === "loss"
      ? "border-red-400/40 bg-red-500/15 text-red-100"
      : "border-amber-400/40 bg-amber-500/15 text-amber-100";

  const movePairs = [];
  for (let i = 0; i < moveHistory.length; i += 2) {
    movePairs.push({ white: moveHistory[i], black: moveHistory[i + 1] || "" });
  }
  const lastPly = moveHistory.length - 1;

  function stripProps(color) {
    const isPlayer = color === playerColor;
    const advantage = color === "w" ? material.balance : -material.balance;
    return {
      color,
      name: isPlayer ? "You" : isMultiplayer ? "Opponent" : ENGINE_NAME,
      detail: !isPlayer && !isMultiplayer ? `Level ${depth}` : "",
      isEngine: !isPlayer && !isMultiplayer,
      active: !gameOver && game.turn() === color,
      thinking: !isPlayer && !isMultiplayer && engineThinking,
      taken: material.takenBy[color],
      advantage,
    };
  }

  const engineBadge = (() => {
    if (isMultiplayer) {
      const live = roomStatus === "open";
      return { dot: live ? "bg-emerald-400" : "bg-amber-400", text: live ? "Room live" : roomStatus === "closed" ? "Room offline" : "Connecting", spin: roomStatus === "connecting" };
    }
    if (engineWarmupStatus === "ready") return { dot: "bg-emerald-400", text: "Engine online" };
    if (engineWarmupStatus === "waking") return { dot: "bg-amber-400", text: `Waking up · ${warmupSeconds}s`, spin: true };
    if (engineWarmupStatus === "error") return { dot: "bg-red-400", text: "Engine offline · retry", retry: true };
    return { dot: "bg-zinc-500", text: "Engine" };
  })();

  return (
    <div className="min-h-dvh bg-zinc-950 text-zinc-100 antialiased" style={{ paddingBottom: "env(safe-area-inset-bottom)" }}>
      <header className="mx-auto flex max-w-6xl items-center justify-between gap-3 px-3 pb-2 pt-3 sm:px-6 lg:pt-5" style={{ paddingTop: "max(0.75rem, env(safe-area-inset-top))" }}>
        <div className="flex min-w-0 items-center gap-2.5">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-emerald-500/15 text-xl text-emerald-300">♞{"︎"}</div>
          <div className="min-w-0">
            <div className="text-base font-semibold leading-tight">{ENGINE_NAME}</div>
            <div className="truncate text-xs text-zinc-500">Neural chess engine</div>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <button
            type="button"
            onClick={() => engineBadge.retry && void warmupEngineServer()}
            className={`flex h-9 items-center gap-2 rounded-full border border-zinc-800 bg-zinc-900 px-3 text-xs font-medium text-zinc-300 ${engineBadge.retry ? "hover:border-zinc-600" : "cursor-default"}`}
            title={engineWarmupStatus === "waking" ? "The free server sleeps when idle and can take up to a minute to wake up." : undefined}
          >
            {engineBadge.spin ? <Loader2 className="h-3.5 w-3.5 animate-spin text-amber-300" /> : <span className={`h-2 w-2 rounded-full ${engineBadge.dot}`} />}
            <span className="whitespace-nowrap">{engineBadge.text}</span>
          </button>
          <button
            type="button"
            onClick={() => setSoundOn((on) => !on)}
            aria-label={soundOn ? "Mute sounds" : "Unmute sounds"}
            className="flex h-9 w-9 items-center justify-center rounded-full border border-zinc-800 bg-zinc-900 text-zinc-400 transition-colors hover:border-zinc-600 hover:text-zinc-100"
          >
            {soundOn ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
          </button>
        </div>
      </header>

      <main className="mx-auto grid max-w-6xl gap-4 px-3 pb-6 sm:px-6 lg:grid-cols-[minmax(0,1fr)_360px] lg:gap-6">
        <section className="min-w-0">
          <div ref={boardAreaRef} className="w-full">
            <div className="mx-auto space-y-2" style={{ width: boardWidth }}>
              <PlayerStrip {...stripProps(topColor)} />

              <div className="relative touch-none select-none" style={{ width: boardWidth, height: boardWidth }}>
                <Chessboard
                  id="HybridBoard"
                  position={fen}
                  boardWidth={boardWidth}
                  onPieceDrop={onDrop}
                  onPieceDragBegin={(piece, square) => {
                    setMoveFrom(square);
                    getMoveOptions(square);
                  }}
                  onPieceDragEnd={clearSelection}
                  onSquareClick={onSquareClick}
                  boardOrientation={boardOrientation}
                  customSquareStyles={customSquareStyles}
                  arePiecesDraggable={canMove}
                  isDraggablePiece={({ piece }) => canMove && piece[0] === playerColor}
                  animationDuration={200}
                  customDarkSquareStyle={{ backgroundColor: "#769656" }}
                  customLightSquareStyle={{ backgroundColor: "#eeeed2" }}
                  customBoardStyle={{ borderRadius: "10px", boxShadow: "0 12px 32px rgba(0,0,0,0.45)" }}
                />

                {engineThinking && engineWaking && (
                  <div className="absolute inset-0 z-40 flex flex-col items-center justify-center gap-3 rounded-[10px] bg-zinc-950/80 p-6 text-center backdrop-blur-sm">
                    <Loader2 className="h-9 w-9 animate-spin text-emerald-300" />
                    <div>
                      <div className="text-base font-semibold text-zinc-100">Waking up {ENGINE_NAME}… {warmupSeconds}s</div>
                      <div className="mt-1 max-w-xs text-sm text-zinc-400">The free server sleeps when nobody plays. The first move can take up to a minute, then moves are quick.</div>
                    </div>
                  </div>
                )}

                {gameEndInfo && !endDismissed && (
                  <div className="absolute inset-0 z-40 flex items-center justify-center rounded-[10px] bg-zinc-950/70 p-4 text-center backdrop-blur-[2px]">
                    <div className={`w-full max-w-xs rounded-2xl border p-5 shadow-2xl ${endTheme}`}>
                      <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-2xl bg-white/10">
                        <EndIcon icon={gameEndInfo.icon} className="h-7 w-7" />
                      </div>
                      <div className="text-2xl font-bold">{gameEndInfo.title}</div>
                      <div className="mt-1.5 text-sm leading-6 opacity-90">{gameEndInfo.reason}</div>
                      <div className="mt-4 grid grid-cols-2 gap-2">
                        <ActionButton className="bg-white/10 text-current hover:bg-white/20" onClick={() => setEndDismissed(true)}>
                          View board
                        </ActionButton>
                        {isMultiplayer ? (
                          <ActionButton className="bg-zinc-50 font-semibold text-zinc-950 hover:bg-white" onClick={leaveRoom}>
                            Leave room
                          </ActionButton>
                        ) : (
                          <ActionButton className="bg-zinc-50 font-semibold text-zinc-950 hover:bg-white" onClick={() => newGame()}>
                            Play again
                          </ActionButton>
                        )}
                      </div>
                    </div>
                  </div>
                )}

                {promotionMoveDetail && (
                  <div className="absolute inset-0 z-50 flex items-center justify-center rounded-[10px] bg-black/55" onClick={() => setPromotionMoveDetail(null)}>
                    <div className="rounded-2xl border border-zinc-700 bg-zinc-900 p-3 shadow-2xl" onClick={(e) => e.stopPropagation()}>
                      <div className="mb-2 text-center text-xs font-medium uppercase tracking-wider text-zinc-400">Promote to</div>
                      <div className="flex gap-2">
                        {["q", "r", "b", "n"].map((piece) => (
                          <button
                            key={piece}
                            type="button"
                            aria-label={{ q: "Queen", r: "Rook", b: "Bishop", n: "Knight" }[piece]}
                            className={`flex h-16 w-16 items-center justify-center rounded-xl text-5xl leading-none transition-transform hover:scale-105 active:scale-95 ${
                              playerColor === "w" ? "bg-zinc-700 text-white hover:bg-zinc-600" : "bg-zinc-200 text-zinc-950 hover:bg-white"
                            }`}
                            onClick={() => handlePromotionSelect(piece)}
                          >
                            {PIECE_GLYPHS[piece]}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <PlayerStrip {...stripProps(bottomColor)} />
            </div>
          </div>

          <div
            ref={mobileMovesRef}
            className="no-scrollbar mx-auto mt-3 flex items-center gap-1 overflow-x-auto whitespace-nowrap rounded-xl bg-zinc-900/80 px-2 py-2 text-sm lg:hidden"
            style={{ maxWidth: boardWidth }}
          >
            {movePairs.length === 0 ? (
              <span className="px-1 text-zinc-500">Moves will appear here</span>
            ) : (
              movePairs.map((pair, index) => (
                <span key={index} className="flex items-center gap-1">
                  <span className="pl-1 text-zinc-600">{index + 1}.</span>
                  <span className={`rounded-md px-1.5 py-0.5 ${index * 2 === lastPly ? "bg-zinc-700 text-white" : "text-zinc-300"}`}>{pair.white}</span>
                  {pair.black && <span className={`rounded-md px-1.5 py-0.5 ${index * 2 + 1 === lastPly ? "bg-zinc-700 text-white" : "text-zinc-300"}`}>{pair.black}</span>}
                </span>
              ))
            )}
          </div>
        </section>

        <aside className="mx-auto w-full min-w-0 max-w-xl space-y-4 lg:max-w-none">
          <Panel>
            <div className="flex items-center gap-2 text-base font-semibold" aria-live="polite">
              {(engineThinking || (isMultiplayer && roomStatus === "connecting")) && <Loader2 className="h-4 w-4 animate-spin text-emerald-300" />}
              <span>{status}</span>
            </div>
            {gameEndInfo && <div className="mt-1 text-sm text-zinc-400">{gameEndInfo.reason}</div>}

            {notice && (
              <div
                className={`mt-3 flex items-center gap-2 rounded-xl border px-3 py-2 text-sm ${
                  notice.tone === "error" ? "border-red-500/30 bg-red-500/10 text-red-200" : "border-sky-500/30 bg-sky-500/10 text-sky-200"
                }`}
              >
                <span className="min-w-0 flex-1 break-words">{notice.text}</span>
                {notice.retry && !engineThinking && (
                  <button type="button" className="shrink-0 rounded-lg bg-red-500/20 px-2.5 py-1 font-medium hover:bg-red-500/30" onClick={() => void maybePlayEngine()}>
                    Try again
                  </button>
                )}
                <button type="button" aria-label="Dismiss" className="shrink-0 rounded-md p-1 opacity-70 hover:opacity-100" onClick={() => setNotice(null)}>
                  <X className="h-4 w-4" />
                </button>
              </div>
            )}

            <div className="mt-4 grid grid-cols-3 gap-2">
              <ActionButton
                className={confirmNewGame ? "bg-amber-400 text-zinc-950 hover:bg-amber-300" : "bg-zinc-100 text-zinc-900 hover:bg-white"}
                disabled={isMultiplayer}
                onClick={requestNewGame}
              >
                <RotateCcw className="h-4 w-4" />
                {confirmNewGame ? "Sure?" : "New"}
              </ActionButton>
              <ActionButton className="bg-zinc-800 text-zinc-100 hover:bg-zinc-700" disabled={isMultiplayer || !playerHasMoved} onClick={takeBack}>
                <Undo2 className="h-4 w-4" /> Undo
              </ActionButton>
              <ActionButton className="bg-zinc-800 text-zinc-100 hover:bg-zinc-700" onClick={() => setFlipped((value) => !value)}>
                <ArrowUpDown className="h-4 w-4" /> Flip
              </ActionButton>
            </div>
          </Panel>

          {!isMultiplayer && (
            <Panel title="Game setup">
              <div className="space-y-4">
                <div>
                  <div className="mb-1.5 flex items-baseline justify-between text-sm">
                    <span className="text-zinc-300">Play as</span>
                    {playerHasMoved && !gameOver && <span className="text-xs text-zinc-500">applies to your next game</span>}
                  </div>
                  <Segmented
                    label="Play as"
                    value={nextColor}
                    onChange={chooseColor}
                    options={[
                      { value: "w", label: "White" },
                      { value: "b", label: "Black" },
                      { value: "random", label: "Random" },
                    ]}
                  />
                </div>
                <div>
                  <div className="mb-1.5 flex items-baseline justify-between text-sm">
                    <span className="text-zinc-300">Engine level</span>
                    <span className="text-xs text-zinc-500">higher = deeper search</span>
                  </div>
                  <Segmented label="Engine level" value={depth} onChange={setDepth} disabled={engineThinking} options={LEVELS.map((level) => ({ value: level, label: level }))} />
                </div>
              </div>
            </Panel>
          )}

          <Panel title="Moves" className="hidden lg:block">
            <div ref={desktopMovesRef} className="max-h-64 overflow-y-auto rounded-xl bg-zinc-950 ring-1 ring-zinc-800">
              {movePairs.length === 0 ? (
                <div className="p-4 text-center text-sm text-zinc-500">No moves yet</div>
              ) : (
                movePairs.map((pair, index) => (
                  <div key={index} className={`grid grid-cols-[2.5rem_1fr_1fr] px-3 py-1 text-sm ${index % 2 ? "bg-zinc-900/40" : ""}`}>
                    <span className="text-zinc-600">{index + 1}.</span>
                    <span className={index * 2 === lastPly ? "font-semibold text-white" : "text-zinc-300"}>{pair.white}</span>
                    <span className={index * 2 + 1 === lastPly ? "font-semibold text-white" : "text-zinc-300"}>{pair.black}</span>
                  </div>
                ))
              )}
            </div>
          </Panel>

          <Panel title="Play a friend" icon={<Users className="h-3.5 w-3.5" />}>
            {isMultiplayer ? (
              <div className="space-y-3">
                <div className="flex items-center justify-between gap-3 rounded-xl bg-zinc-950 px-3 py-2.5 ring-1 ring-zinc-800">
                  <div>
                    <div className="text-xs text-zinc-500">Room code · you play {playerColor === "w" ? "White" : "Black"}</div>
                    <div className="font-mono text-lg font-semibold tracking-widest text-white">{roomId}</div>
                  </div>
                  <span className={`h-2.5 w-2.5 rounded-full ${roomStatus === "open" ? "bg-emerald-400" : roomStatus === "connecting" ? "bg-amber-400" : "bg-red-400"}`} />
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <ActionButton className="bg-emerald-600 text-white hover:bg-emerald-500" onClick={shareRoom}>
                    {copied ? <Check className="h-4 w-4" /> : navigator.share ? <Share2 className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                    {copied ? "Link copied" : "Invite link"}
                  </ActionButton>
                  <ActionButton className="bg-zinc-800 text-zinc-100 hover:bg-zinc-700" onClick={leaveRoom}>
                    <X className="h-4 w-4" /> Leave
                  </ActionButton>
                </div>
              </div>
            ) : joinOpen ? (
              <form
                className="flex gap-2"
                onSubmit={(event) => {
                  event.preventDefault();
                  joinRoom(joinCode);
                }}
              >
                <input
                  autoFocus
                  value={joinCode}
                  onChange={(event) => setJoinCode(event.target.value)}
                  placeholder="Room code"
                  autoCapitalize="none"
                  autoCorrect="off"
                  spellCheck={false}
                  className="h-11 min-w-0 flex-1 rounded-xl bg-zinc-950 px-3 font-mono text-base text-white ring-1 ring-zinc-700 placeholder:font-sans placeholder:text-zinc-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
                <ActionButton type="submit" className="bg-indigo-600 px-4 text-white hover:bg-indigo-500" disabled={!joinCode.trim()}>
                  Join
                </ActionButton>
                <ActionButton className="w-11 bg-zinc-800 text-zinc-300 hover:bg-zinc-700" aria-label="Cancel" onClick={() => setJoinOpen(false)}>
                  <X className="h-4 w-4" />
                </ActionButton>
              </form>
            ) : (
              <div className="grid grid-cols-2 gap-2">
                <ActionButton className="bg-emerald-600 text-white hover:bg-emerald-500" onClick={hostRoom}>
                  <Globe className="h-4 w-4" /> Create room
                </ActionButton>
                <ActionButton className="bg-indigo-600 text-white hover:bg-indigo-500" onClick={() => setJoinOpen(true)}>
                  <LogIn className="h-4 w-4" /> Join room
                </ActionButton>
              </div>
            )}
          </Panel>
        </aside>
      </main>
    </div>
  );
}

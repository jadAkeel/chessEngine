import React, { useMemo, useRef, useState, useEffect } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Brain, CircleEqual, Loader2, RotateCcw, ShieldAlert, Swords, Trophy, Zap, Globe, Users } from "lucide-react";
import { playMoveSoundFor } from "@/utils/sound";

const DEFAULT_API_BASE_URL = import.meta.env.PROD
  ? "https://chessengine-2.onrender.com"
  : "http://localhost:8000";
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL).replace(/\/$/, "");

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

function pieceValue(piece) {
  if (!piece) return 0;
  switch (piece.type) {
    case "p": return 100;
    case "n": return 320;
    case "b": return 330;
    case "r": return 500;
    case "q": return 900;
    default: return 0;
  }
}

function simpleEvaluateMove(game, move) {
  let score = 0;
  const from = move.from;
  const to = move.to;
  const targetFile = to.charCodeAt(0) - 96;
  const targetRank = parseInt(to[1], 10);

  if (move.captured) score += pieceValue({ type: move.captured }) * 10;
  if (move.promotion) score += 800;
  if ([4, 5].includes(targetFile) && [4, 5].includes(targetRank)) score += 25;
  if (move.san && move.san.includes("+")) score += 40;

  game.move(move);
  if (game.isCheckmate()) score += 100000;
  if (game.isCheck()) score += 30;
  game.undo();

  const movingPiece = game.get(from);
  if (movingPiece?.type === "n" || movingPiece?.type === "b") score += 8;

  return score + Math.random() * 3;
}

function pickEngineMove(game, depth) {
  const moves = game.moves({ verbose: true });
  if (!moves.length) return null;

  const scored = moves.map((move) => ({ move, score: simpleEvaluateMove(game, move) + depth * 2 }));
  scored.sort((a, b) => b.score - a.score);
  return scored[0].move;
}

function formatStatus(game, engineThinking, playerColor, isMultiplayer, engineWarmupStatus) {
  if (!isMultiplayer && engineWarmupStatus === "waking" && engineThinking) return "Loading engine model...";
  if (!isMultiplayer && engineWarmupStatus === "waking") return "Preparing engine server...";
  if (!isMultiplayer && engineWarmupStatus === "error" && engineThinking) return "Using local evaluation...";
  if (!isMultiplayer && engineThinking) return "Engine is thinking...";
  if (game.isCheckmate()) {
    return game.turn() === "w" ? "Black wins by checkmate" : "White wins by checkmate";
  }
  if (game.isDraw()) return "Draw";
  const side = game.turn() === "w" ? "White" : "Black";
  const suffix = game.isCheck() ? " is in check" : " to move";
  if (game.turn() === playerColor) {
    return `Your turn • ${side}${suffix}`;
  }
  return `${isMultiplayer ? "Opponent" : "Engine"} turn • ${side}${suffix}`;
}

function CapturedPieces({ game }) {
  const board = game.board();
  const current = { w: { p: 0, n: 0, b: 0, r: 0, q: 0 }, b: { p: 0, n: 0, b: 0, r: 0, q: 0 } };
  for (const row of board) {
    for (const piece of row) {
      if (piece && piece.type !== "k") current[piece.color][piece.type] += 1;
    }
  }
  const maxCounts = { p: 8, n: 2, b: 2, r: 2, q: 1 };
  const whiteCaptured = [];
  const blackCaptured = [];

  Object.keys(maxCounts).forEach((type) => {
    const missingWhite = maxCounts[type] - current.w[type];
    const missingBlack = maxCounts[type] - current.b[type];
    for (let i = 0; i < missingWhite; i++) whiteCaptured.push(type.toUpperCase());
    for (let i = 0; i < missingBlack; i++) blackCaptured.push(type.toUpperCase());
  });

  return (
    <div className="grid grid-cols-2 gap-3 text-sm">
      <div className="rounded-2xl border border-zinc-800 p-3 bg-zinc-950">
        <div className="mb-2 font-medium text-zinc-300">White lost</div>
        <div className="flex min-h-10 flex-wrap gap-1 text-zinc-600">{whiteCaptured.length ? whiteCaptured.map((p, i) => <Badge key={i} variant="secondary" className="bg-zinc-800 text-zinc-300">{p}</Badge>) : <span className="text-zinc-500">None</span>}</div>
      </div>
      <div className="rounded-2xl border border-zinc-800 p-3 bg-zinc-950">
        <div className="mb-2 font-medium text-zinc-300">Black lost</div>
        <div className="flex min-h-10 flex-wrap gap-1 text-zinc-600">{blackCaptured.length ? blackCaptured.map((p, i) => <Badge key={i} variant="secondary" className="bg-zinc-800 text-zinc-300">{p}</Badge>) : <span className="text-zinc-500">None</span>}</div>
      </div>
    </div>
  );
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

function squareFromBoardPosition(rowIndex, colIndex) {
  return `${String.fromCharCode(97 + colIndex)}${8 - rowIndex}`;
}

function findKingSquare(game, color) {
  const board = game.board();
  for (let rowIndex = 0; rowIndex < board.length; rowIndex += 1) {
    for (let colIndex = 0; colIndex < board[rowIndex].length; colIndex += 1) {
      const piece = board[rowIndex][colIndex];
      if (piece?.type === "k" && piece.color === color) {
        return squareFromBoardPosition(rowIndex, colIndex);
      }
    }
  }
  return null;
}

function getDrawReason(game) {
  if (game.isStalemate?.()) return "لا توجد نقلة قانونية والملك ليس في كش.";
  if (game.isInsufficientMaterial?.()) return "لا توجد قطع كافية لفرض كش مات.";
  if (game.isThreefoldRepetition?.()) return "تكرر نفس الوضع ثلاث مرات.";
  if (game.isDrawByFiftyMoves?.()) return "مرّت 50 نقلة بدون أسر أو تحريك بيدق.";
  return "انتهت المباراة بدون فائز حسب قواعد الشطرنج.";
}

function getGameEndInfo(game, playerColor, isMultiplayer) {
  if (!game.isGameOver()) return null;

  if (game.isCheckmate()) {
    const winnerColor = game.turn() === "w" ? "b" : "w";
    const winnerName = winnerColor === "w" ? "الأبيض" : "الأسود";
    const playerWon = winnerColor === playerColor;
    const title = isMultiplayer ? `فاز ${winnerName}` : playerWon ? "فزت بالمباراة" : "خسرت المباراة";

    return {
      tone: playerWon ? "win" : "loss",
      title,
      reason: `السبب: كش مات. ${winnerName} حاصر الملك بدون أي نقلة إنقاذ.`,
      icon: playerWon ? "trophy" : "alert",
    };
  }

  return {
    tone: "draw",
    title: "تعادل",
    reason: `السبب: ${getDrawReason(game)}`,
    icon: "draw",
  };
}

export default function ChessHybridApp() {
  const gameRef = useRef(new Chess());
  const [fen, setFen] = useState(gameRef.current.fen());
  const [moveHistory, setMoveHistory] = useState([]);
  const [playerColor, setPlayerColor] = useState("w");
  const [depth, setDepth] = useState("6");
  const [engineThinking, setEngineThinking] = useState(false);
  const [engineWarmupStatus, setEngineWarmupStatus] = useState("idle");
  const [lastMoveSquares, setLastMoveSquares] = useState({});
  const [lastMoveNote, setLastMoveNote] = useState("");
  const [moveFrom, setMoveFrom] = useState("");
  const [optionSquares, setOptionSquares] = useState({});

  const [isMultiplayer, setIsMultiplayer] = useState(false);
  const [roomId, setRoomId] = useState("");

  const [showPromotionDialog, setShowPromotionDialog] = useState(false);
  const [promotionMoveDetail, setPromotionMoveDetail] = useState(null);

  const wsRef = useRef(null);
  const engineWarmupPromiseRef = useRef(null);
  const game = gameRef.current;
  const playerTurn = game.turn() === playerColor;
  const engineWaking = !isMultiplayer && engineWarmupStatus === "waking";
  const engineLoadOverlayVisible = engineThinking && engineWaking;

  const boardOrientation = playerColor === "w" ? "white" : "black";
  const checkedKingSquare = useMemo(() => {
    if (!game.isCheck()) return null;
    return findKingSquare(game, game.turn());
  }, [fen, game]);
  const gameEndInfo = useMemo(() => getGameEndInfo(game, playerColor, isMultiplayer), [fen, playerColor, isMultiplayer, game]);

  useEffect(() => {
    void warmupEngineServer();
  }, []);

  useEffect(() => {
    if (isMultiplayer && roomId) {
      const wsUrl = buildWebSocketUrl(roomId);

      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "init" || data.type === "move") {
          gameRef.current.load(data.fen);
          syncGame();
        } else if (data.type === "error" && data.fen) {
          gameRef.current.load(data.fen);
          syncGame();
        }
      };

      return () => {
        wsRef.current?.close();
      };
    }
  }, [isMultiplayer, roomId]);

  const customSquareStyles = useMemo(() => {
    const styles = { ...lastMoveSquares, ...optionSquares };
    if (moveFrom) {
      styles[moveFrom] = {
        ...styles[moveFrom],
        background: "rgba(255, 255, 0, 0.4)",
      };
    }
    if (checkedKingSquare) {
      styles[checkedKingSquare] = {
        ...styles[checkedKingSquare],
        background: "linear-gradient(135deg, rgba(220, 38, 38, 0.9), rgba(127, 29, 29, 0.92))",
        boxShadow: "inset 0 0 0 4px rgba(254, 202, 202, 0.95), inset 0 0 22px rgba(248, 113, 113, 0.9)",
      };
    }
    return styles;
  }, [checkedKingSquare, lastMoveSquares, moveFrom, optionSquares]);

  function syncGame() {
    setFen(game.fen());
    setMoveHistory(game.history({ verbose: true }).map(formatHistoryMove));
  }

  function loadAuthoritativeFen(nextFen, context) {
    if (!nextFen) return false;
    try {
      game.load(nextFen);
      syncGame();
      return true;
    } catch (err) {
      console.error("Failed to load authoritative FEN", { context, nextFen, err });
      return false;
    }
  }

  function moveToUci(move) {
    return `${move.from}${move.to}${move.promotion || ""}`;
  }

  function highlightLastMove(move) {
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
      setLastMoveNote(`${formatHistoryMove(move)} captured en passant on ${capturedSquare}`);
    } else {
      setLastMoveNote("");
    }
    setLastMoveSquares(styles);
  }

  function applyEngineMove(uci, authoritativeFen) {
    const from = uci.slice(0, 2);
    const to = uci.slice(2, 4);
    const moveDetails = { from, to };
    if (uci.length === 5) {
      moveDetails.promotion = uci[4];
    } else {
      moveDetails.promotion = "q";
    }

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
      console.warn("Frontend/backend FEN mismatch after engine move", {
        uci,
        localFen: game.fen(),
        authoritativeFen,
      });
      loadAuthoritativeFen(authoritativeFen, "engine-move-mismatch");
    }
  }

  async function warmupEngineServer() {
    if (isMultiplayer) return true;
    if (engineWarmupStatus === "ready") return true;
    if (engineWarmupPromiseRef.current) return engineWarmupPromiseRef.current;

    setEngineWarmupStatus("waking");
    const warmupPromise = fetch(`${API_BASE_URL}/health`, { cache: "no-store" })
      .then(async (res) => {
        if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
        const data = await res.json().catch(() => ({}));
        if (data.model === false) throw new Error("Model is not loaded");
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

    setEngineThinking(true);
    try {
      const serverReady = await warmupEngineServer();
      if (!serverReady) throw new Error("Engine server is not ready");

      const engineDepth = Number(depth);
      const candidateCount = Math.max(10, Math.min(16, engineDepth * 3));
      const requestFen = game.fen();
      const res = await fetch(`${API_BASE_URL}/fastmove`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          fen: requestFen,
          topk: candidateCount,
          depth: engineDepth,
          max_simulations: 96,
          adaptive: true
        })
      });
      if (!res.ok) throw new Error(`Engine request failed: ${res.status}`);
      const data = await res.json();

      const uci = data.move || data.moves?.[0]?.uci || data.best_move;
      if (uci) {
        if (normalizeFen(game.fen()) !== normalizeFen(requestFen)) {
          console.warn("Ignoring stale engine response", {
            uci,
            requestFen,
            currentFen: game.fen(),
          });
          return;
        }
        if (data.fen_before && normalizeFen(data.fen_before) !== normalizeFen(requestFen)) {
          console.warn("Engine response FEN does not match request FEN", {
            uci,
            requestFen,
            responseFen: data.fen_before,
          });
        }
        applyEngineMove(uci, data.fen_after);
      }
    } catch (err) {
      console.error('Engine request failed', err);
      const fallback = pickEngineMove(game, Number(depth));
      if (fallback) {
        const move = game.move(fallback);
        if (move) {
          highlightLastMove(move);
          syncGame();
          playMoveSoundFor(move, game);
        }
      }
    } finally {
      setEngineThinking(false);
    }
  }

  function safeGameMutate(modifier) {
    modifier(game);
    syncGame();
  }

  function getMoveOptions(square) {
    const moves = game.moves({ square, verbose: true });
    if (moves.length === 0) {
      setOptionSquares({});
      return;
    }
    const options = {};
    moves.forEach((m) => {
      options[m.to] = {
        background: game.get(m.to)
          ? "radial-gradient(circle, transparent 60%, rgba(0,0,0,.25) 61%, rgba(0,0,0,.25) 80%, transparent 81%)"
          : "radial-gradient(circle, rgba(0,0,0,.25) 23%, transparent 24%)",
      };
    });
    setOptionSquares(options);
  }

  function onDrop(sourceSquare, targetSquare, piece) {
    if (isMultiplayer) {
      if (game.turn() !== playerColor) return false;
    } else {
      if (engineThinking || !playerTurn) return false;
    }

    let adjustedTarget = targetSquare;
    const sourcePiece = game.get(sourceSquare);
    const targetPiece = game.get(targetSquare);
    if (sourcePiece?.type === 'k' && targetPiece?.type === 'r' && sourcePiece.color === targetPiece.color) {
      if (sourceSquare === 'e1' && targetSquare === 'h1') adjustedTarget = 'g1';
      if (sourceSquare === 'e1' && targetSquare === 'a1') adjustedTarget = 'c1';
      if (sourceSquare === 'e8' && targetSquare === 'h8') adjustedTarget = 'g8';
      if (sourceSquare === 'e8' && targetSquare === 'a8') adjustedTarget = 'c8';
    }

    const isPawnMove = piece?.toLowerCase().includes("p");
    const isPromotionRank = adjustedTarget[1] === "8" || adjustedTarget[1] === "1";
    
    if (isPawnMove && isPromotionRank) {
      setPromotionMoveDetail({ from: sourceSquare, to: adjustedTarget });
      setShowPromotionDialog(true);
      return true;
    }

    try {
      const move = game.move({
        from: sourceSquare,
        to: adjustedTarget,
      });
      setMoveFrom("");
      setOptionSquares({});
      highlightLastMove(move);
      syncGame();
      playMoveSoundFor(move, game);

      if (isMultiplayer) {
        wsRef.current?.send(JSON.stringify({ type: "move", move: moveToUci(move) }));
      } else {
        void maybePlayEngine();
      }
      return true;
    } catch (e) {
      return false;
    }
  }

  function handlePromotionSelect(promotionPieceType) {
    setShowPromotionDialog(false);
    if (!promotionMoveDetail) return;

    try {
      const move = game.move({
        from: promotionMoveDetail.from,
        to: promotionMoveDetail.to,
        promotion: promotionPieceType
      });
      setPromotionMoveDetail(null);
      setMoveFrom("");
      setOptionSquares({});
      highlightLastMove(move);
      syncGame();
      playMoveSoundFor(move, game);
      
      if (isMultiplayer) {
        wsRef.current?.send(JSON.stringify({ type: "move", move: moveToUci(move) }));
      } else {
        void maybePlayEngine();
      }
    } catch (e) {
      setPromotionMoveDetail(null);
    }
  }

  function onSquareClick(square) {
    if (isMultiplayer) {
      if (game.turn() !== playerColor) return;
    } else {
      if (engineThinking || !playerTurn) return;
    }

    if (!moveFrom) {
      const piece = game.get(square);
      if (piece && piece.color === playerColor) {
        setMoveFrom(square);
        getMoveOptions(square);
      }
      return;
    }

    try {
      const moves = game.moves({ square: moveFrom, verbose: true });
      
      let adjustedTarget = square;
      const sourcePiece = game.get(moveFrom);
      const targetPiece = game.get(square);
      if (sourcePiece?.type === 'k' && targetPiece?.type === 'r' && sourcePiece.color === targetPiece.color) {
        if (moveFrom === 'e1' && square === 'h1') adjustedTarget = 'g1';
        if (moveFrom === 'e1' && square === 'a1') adjustedTarget = 'c1';
        if (moveFrom === 'e8' && square === 'h8') adjustedTarget = 'g8';
        if (moveFrom === 'e8' && square === 'a8') adjustedTarget = 'c8';
      }

      const validMove = moves.find(m => m.to === adjustedTarget);

      if (validMove) {
        const piece = game.get(moveFrom);
        const isPawnMove = piece?.type?.toLowerCase() === "p";
        const isPromotionRank = adjustedTarget[1] === "8" || adjustedTarget[1] === "1";

        if (isPawnMove && isPromotionRank) {
          setPromotionMoveDetail({ from: moveFrom, to: adjustedTarget });
          setShowPromotionDialog(true);
          return;
        }

        const move = game.move({
          from: moveFrom,
          to: adjustedTarget,
        });
        setMoveFrom("");
        setOptionSquares({});
        highlightLastMove(move);
        syncGame();
        playMoveSoundFor(move, game);

        if (isMultiplayer) {
          wsRef.current?.send(JSON.stringify({ type: "move", move: moveToUci(move) }));
        } else {
          void maybePlayEngine();
        }
      } else {
         const piece = game.get(square);
         if (piece && piece.color === playerColor) {
           setMoveFrom(square);
           getMoveOptions(square);
         } else {
           setMoveFrom("");
           setOptionSquares({});
         }
      }
    } catch {
      const piece = game.get(square);
      if (piece && piece.color === playerColor) {
        setMoveFrom(square);
        getMoveOptions(square);
      } else {
        setMoveFrom("");
        setOptionSquares({});
      }
    }
  }

  function newGame(nextColor = playerColor) {
    safeGameMutate((g) => g.reset());
    setLastMoveSquares({});
    setLastMoveNote("");
    setMoveFrom("");
    setOptionSquares({});
    setPlayerColor(nextColor);
    if (nextColor === "b") {
      void maybePlayEngine(nextColor);
    }
  }

  function startMultiplayer() {
    setIsMultiplayer(true);
    const newRoomId = Math.random().toString(36).substring(2, 7);
    setRoomId(newRoomId);
    safeGameMutate((g) => g.reset());
  }

  function joinMultiplayer() {
    const inputRoom = prompt("Enter Room ID to join:");
    if (inputRoom) {
      setIsMultiplayer(true);
      setRoomId(inputRoom);
      setPlayerColor("b");
    }
  }

  const status = formatStatus(game, engineThinking, playerColor, isMultiplayer, engineWarmupStatus);
  const activeSideLabel = game.turn() === "w" ? "White" : "Black";
  const gameModeLabel = isMultiplayer ? "Online" : "Vs AI";
  const gameEndTheme = gameEndInfo?.tone === "win"
    ? "border-emerald-400/40 bg-emerald-500/15 text-emerald-100"
    : gameEndInfo?.tone === "loss"
      ? "border-red-400/40 bg-red-500/15 text-red-100"
      : "border-amber-400/40 bg-amber-500/15 text-amber-100";

  const movePairs = [];
  for (let i = 0; i < moveHistory.length; i += 2) {
    movePairs.push({ white: moveHistory[i], black: moveHistory[i + 1] || "" });
  }

  return (
    <div className="min-h-screen overflow-hidden bg-zinc-950 text-zinc-100 font-sans">
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(circle_at_top_left,rgba(16,185,129,0.16),transparent_34%),radial-gradient(circle_at_bottom_right,rgba(99,102,241,0.14),transparent_32%)]" />
      <div className="relative mx-auto grid max-w-7xl gap-4 p-3 sm:gap-6 sm:p-6 lg:grid-cols-[1.2fr_0.8fr]">
        <Card className="rounded-[2rem] border border-zinc-800/80 bg-zinc-900/90 p-3 shadow-2xl backdrop-blur sm:p-4">
          <CardHeader className="flex flex-col gap-3 pb-4 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <CardTitle className="text-xl font-semibold text-zinc-100 sm:text-2xl">Hybrid Chess Arena</CardTitle>
              <p className="mt-1 text-sm text-zinc-400">Mobile-ready chess arena with Python backend.</p>
            </div>
            <div className="flex flex-wrap gap-2">
              <Badge className="rounded-full bg-emerald-500/15 px-3 py-1 text-emerald-300">Mobile ready</Badge>
              {checkedKingSquare && <Badge className="rounded-full bg-red-500/15 px-3 py-1 text-red-200">Check on {checkedKingSquare}</Badge>}
            </div>
          </CardHeader>
          <CardContent>
            <div className="mb-3 grid grid-cols-3 gap-2 sm:hidden">
              <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3">
                <div className="text-[11px] uppercase tracking-wide text-zinc-500">Turn</div>
                <div className="mt-1 text-sm font-semibold text-zinc-100">{activeSideLabel}</div>
              </div>
              <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3">
                <div className="text-[11px] uppercase tracking-wide text-zinc-500">Mode</div>
                <div className="mt-1 text-sm font-semibold text-zinc-100">{gameModeLabel}</div>
              </div>
              <div className={`rounded-2xl border p-3 ${checkedKingSquare ? "border-red-500/40 bg-red-500/15" : "border-zinc-800 bg-zinc-950/80"}`}>
                <div className="text-[11px] uppercase tracking-wide text-zinc-500">King</div>
                <div className={`mt-1 text-sm font-semibold ${checkedKingSquare ? "text-red-100" : "text-zinc-100"}`}>{checkedKingSquare ? "Check" : "Safe"}</div>
              </div>
            </div>
            <div className="rounded-[1.75rem] border border-zinc-800 bg-zinc-950/90 p-2 shadow-inner sm:p-4 flex justify-center">
              <div className="w-full max-w-[600px] relative">
                <Chessboard
                  id="HybridBoard"
                  position={fen}
                  onPieceDrop={onDrop}
                  onPieceDragBegin={(piece, square) => {
                    setMoveFrom(square);
                    getMoveOptions(square);
                  }}
                  onPieceDragEnd={() => {
                    setMoveFrom("");
                    setOptionSquares({});
                  }}
                  onSquareClick={onSquareClick}
                  boardOrientation={boardOrientation}
                  customSquareStyles={customSquareStyles}
                  arePiecesDraggable={!engineThinking && !game.isGameOver() && !(engineWaking && !playerTurn)}
                  customDarkSquareStyle={{ backgroundColor: "#769656" }}
                  customLightSquareStyle={{ backgroundColor: "#eeeed2" }}
                  customBoardStyle={{ borderRadius: "18px", boxShadow: "0 18px 45px rgba(0,0,0,0.42)", overflow: "hidden" }}
                />

                {engineLoadOverlayVisible && (
                  <div className="absolute inset-0 z-40 flex flex-col items-center justify-center gap-3 rounded-[18px] bg-zinc-950/80 text-center backdrop-blur-sm">
                    <Loader2 className="h-9 w-9 animate-spin text-emerald-300" />
                    <div>
                      <div className="text-base font-semibold text-zinc-100">Loading engine model...</div>
                      <div className="mt-1 text-sm text-zinc-400">The first move can take a moment on Render.</div>
                    </div>
                  </div>
                )}

                {gameEndInfo && (
                  <div className="absolute inset-0 z-40 flex items-center justify-center rounded-[18px] bg-zinc-950/78 p-4 text-center backdrop-blur-sm">
                    <div className={`w-full max-w-sm rounded-3xl border p-5 shadow-2xl ${gameEndTheme}`}>
                      <div className="mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-2xl bg-white/10">
                        {gameEndInfo.icon === "draw" ? <CircleEqual className="h-8 w-8" /> : gameEndInfo.icon === "trophy" ? <Trophy className="h-8 w-8" /> : <ShieldAlert className="h-8 w-8" />}
                      </div>
                      <div className="text-2xl font-bold">{gameEndInfo.title}</div>
                      <div className="mt-2 text-sm leading-6 opacity-90">{gameEndInfo.reason}</div>
                      <Button className="mt-5 w-full rounded-2xl bg-zinc-50 px-4 py-3 font-semibold text-zinc-950 hover:bg-white" onClick={() => newGame(playerColor)}>
                        Play again
                      </Button>
                    </div>
                  </div>
                )}

                {showPromotionDialog && (
                  <div className="absolute inset-0 z-50 flex items-center justify-center rounded-[18px] bg-black/50">
                    <div className="bg-zinc-800 p-4 rounded-xl flex gap-2 shadow-2xl border border-zinc-700">
                      {["q", "r", "n", "b"].map((piece) => (
                        <button
                          key={piece}
                          className="w-14 h-14 bg-zinc-900 hover:bg-zinc-700 rounded-lg flex items-center justify-center transition-colors"
                          onClick={() => handlePromotionSelect(piece)}
                        >
                          <img 
                            src={`https://www.chess.com/chess-themes/pieces/neo/150/${playerColor}${piece}.png`} 
                            alt={piece} 
                            className="w-12 h-12" 
                          />
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="grid gap-4 sm:gap-6">
          <Card className="rounded-[2rem] border border-zinc-800/80 bg-zinc-900/90 p-3 shadow-xl backdrop-blur sm:p-4">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-lg text-zinc-100"><Brain className="h-5 w-5" /> Game Status</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center gap-2 rounded-2xl border border-zinc-800 bg-zinc-950/90 px-4 py-3 text-sm font-medium text-zinc-300">
                {engineWaking && <Loader2 className="h-4 w-4 animate-spin text-emerald-300" />}
                <span>{status}</span>
              </div>
              {gameEndInfo && (
                <div className={`rounded-3xl border p-4 ${gameEndTheme}`}>
                  <div className="flex items-start gap-3">
                    <div className="mt-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-white/10">
                      {gameEndInfo.icon === "draw" ? <CircleEqual className="h-5 w-5" /> : gameEndInfo.icon === "trophy" ? <Trophy className="h-5 w-5" /> : <ShieldAlert className="h-5 w-5" />}
                    </div>
                    <div>
                      <div className="font-bold">{gameEndInfo.title}</div>
                      <div className="mt-1 text-sm leading-6 opacity-90">{gameEndInfo.reason}</div>
                    </div>
                  </div>
                </div>
              )}
              {lastMoveNote && (
                <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-2 text-sm font-medium text-red-200">
                  {lastMoveNote}
                </div>
              )}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <div className="mb-2 text-sm text-zinc-400 pl-1">Play as</div>
                  <Select value={playerColor} onValueChange={(value) => newGame(value)} disabled={engineThinking}>
                    <SelectTrigger className="w-full justify-between rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-200">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="w">White</SelectItem>
                      <SelectItem value="b">Black</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <div className="mb-2 text-sm text-zinc-400 pl-1">Depth</div>
                  <Select value={depth} onValueChange={setDepth} disabled={engineThinking}>
                    <SelectTrigger className="w-full justify-between rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-200">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1">1</SelectItem>
                      <SelectItem value="2">2</SelectItem>
                      <SelectItem value="3">3</SelectItem>
                      <SelectItem value="4">4</SelectItem>
                      <SelectItem value="5">5</SelectItem>
                      <SelectItem value="6">6</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="flex flex-col gap-3 pt-2 sm:flex-row">
                <Button className="flex-1 rounded-xl bg-zinc-100 text-zinc-900 hover:bg-zinc-200 py-2.5 flex justify-center items-center font-medium disabled:cursor-not-allowed disabled:opacity-60" disabled={engineThinking} onClick={() => {
                  setIsMultiplayer(false);
                  newGame(playerColor);
                }}>
                  <RotateCcw className="mr-2 h-4 w-4" /> New AI Game
                </Button>
                <Button className="flex-1 rounded-xl bg-zinc-800 text-zinc-100 hover:bg-zinc-700 py-2.5 flex justify-center items-center font-medium disabled:cursor-not-allowed disabled:opacity-60" onClick={maybePlayEngine} disabled={engineThinking || isMultiplayer || engineWaking || game.isGameOver()}>
                  <Zap className="mr-2 h-4 w-4" /> AI Move
                </Button>
              </div>

              <div className="flex flex-col gap-3 border-t border-zinc-800 pt-2 sm:flex-row">
                {!isMultiplayer ? (
                  <>
                    <Button className="flex-1 rounded-xl bg-emerald-600 text-white hover:bg-emerald-500 py-2.5 disabled:cursor-not-allowed disabled:opacity-60" disabled={engineThinking} onClick={startMultiplayer}>
                      <Globe className="mr-2 h-4 w-4" /> Host online
                    </Button>
                    <Button className="flex-1 rounded-xl bg-indigo-600 text-white hover:bg-indigo-500 py-2.5 disabled:cursor-not-allowed disabled:opacity-60" disabled={engineThinking} onClick={joinMultiplayer}>
                      <Users className="mr-2 h-4 w-4" /> Join 
                    </Button>
                  </>
                ) : (
                  <div className="flex w-full items-center justify-between gap-3 rounded-xl border border-zinc-700 bg-zinc-800/50 px-4 py-2">
                    <span className="text-zinc-300 font-medium">Room: <span className="text-white font-mono">{roomId}</span></span>
                    <Button variant="outline" size="sm" className="h-7 bg-zinc-900 text-xs border-zinc-600" onClick={() => setIsMultiplayer(false)}>Quit</Button>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          <Card className="rounded-[2rem] border border-zinc-800/80 bg-zinc-900/90 p-3 shadow-xl backdrop-blur sm:p-4">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-lg text-zinc-100"><Swords className="h-5 w-5" /> Move List</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="max-h-[220px] overflow-auto rounded-2xl border border-zinc-800 bg-zinc-950/90 custom-scrollbar">
                <div className="grid grid-cols-[50px_1fr_1fr] gap-x-3 border-b border-zinc-800 px-4 py-2.5 text-xs uppercase tracking-wider text-zinc-500 font-medium sticky top-0 bg-zinc-950/95 backdrop-blur">
                  <div>#</div>
                  <div>White</div>
                  <div>Black</div>
                </div>
                {movePairs.length === 0 ? (
                  <div className="p-4 text-sm text-zinc-500 text-center italic">No moves yet.</div>
                ) : (
                  <div className="py-1">
                    {movePairs.map((pair, index) => (
                      <div key={index} className="grid grid-cols-[50px_1fr_1fr] gap-x-3 px-4 py-1.5 text-sm hover:bg-zinc-900/50 transition-colors">
                        <div className="text-zinc-600 select-none">{index + 1}.</div>
                        <div className="text-zinc-300">{pair.white}</div>
                        <div className="text-zinc-400">{pair.black}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          <Card className="rounded-[2rem] border border-zinc-800/80 bg-zinc-900/90 p-3 shadow-xl backdrop-blur sm:p-4">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg text-zinc-100">Captured Pieces</CardTitle>
            </CardHeader>
            <CardContent>
              <CapturedPieces game={game} />
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

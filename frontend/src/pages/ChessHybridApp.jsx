import React, { useMemo, useRef, useState, useEffect } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Brain, RotateCcw, Swords, Zap, Globe, Users } from "lucide-react";
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

function formatStatus(game, engineThinking, playerColor, isMultiplayer) {
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

export default function ChessHybridApp() {
  const gameRef = useRef(new Chess());
  const [fen, setFen] = useState(gameRef.current.fen());
  const [moveHistory, setMoveHistory] = useState([]);
  const [playerColor, setPlayerColor] = useState("w");
  const [depth, setDepth] = useState("3");
  const [engineThinking, setEngineThinking] = useState(false);
  const [lastMoveSquares, setLastMoveSquares] = useState({});
  const [moveFrom, setMoveFrom] = useState("");
  const [optionSquares, setOptionSquares] = useState({});

  const [isMultiplayer, setIsMultiplayer] = useState(false);
  const [roomId, setRoomId] = useState("");

  const [showPromotionDialog, setShowPromotionDialog] = useState(false);
  const [promotionMoveDetail, setPromotionMoveDetail] = useState(null);

  const wsRef = useRef(null);
  const game = gameRef.current;
  const playerTurn = game.turn() === playerColor;

  const boardOrientation = playerColor === "w" ? "white" : "black";

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
    return styles;
  }, [lastMoveSquares, moveFrom, optionSquares]);

  function syncGame() {
    setFen(game.fen());
    setMoveHistory([...game.history()]);
  }

  function moveToUci(move) {
    return `${move.from}${move.to}${move.promotion || ""}`;
  }

  function highlightLastMove(from, to) {
    setLastMoveSquares({
      [from]: { background: "rgba(250, 204, 21, 0.35)" },
      [to]: { background: "rgba(250, 204, 21, 0.35)" },
    });
  }

  async function maybePlayEngine(activePlayerColor = playerColor) {
    if (game.isGameOver()) return;
    if (game.turn() === activePlayerColor) return;

    setEngineThinking(true);
    try {
      const res = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen: game.fen(), topk: 1 })
      });
      const data = await res.json();

      const uci = data.moves?.[0]?.uci || data.move || data.best_move;
      if (uci) {
        const from = uci.slice(0, 2);
        const to = uci.slice(2, 4);
        
        const moveDetails = { from, to };
        if (uci.length === 5) {
          moveDetails.promotion = uci[4];
        } else {
          moveDetails.promotion = 'q';
        }

        const move = game.move(moveDetails);
        if (move) {
          highlightLastMove(move.from, move.to);
          syncGame();
          playMoveSoundFor(move, game);
        }
      }
    } catch (err) {
      console.error('Engine request failed', err);
      const fallback = pickEngineMove(game, Number(depth));
      if (fallback) {
        game.move(fallback);
        highlightLastMove(fallback.from, fallback.to);
        syncGame();
        playMoveSoundFor(fallback, game);
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
      highlightLastMove(move.from, move.to);
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
      highlightLastMove(move.from, move.to);
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
        highlightLastMove(move.from, move.to);
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

  const status = formatStatus(game, engineThinking, playerColor, isMultiplayer);

  const movePairs = [];
  for (let i = 0; i < moveHistory.length; i += 2) {
    movePairs.push({ white: moveHistory[i], black: moveHistory[i + 1] || "" });
  }

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 font-sans">
      <div className="mx-auto grid max-w-7xl gap-6 p-6 lg:grid-cols-[1.2fr_0.8fr]">
        <Card className="rounded-3xl border-zinc-800 bg-zinc-900 shadow-2xl p-4">
          <CardHeader className="flex flex-row items-center justify-between pb-4">
            <div>
              <CardTitle className="text-2xl font-semibold text-zinc-100">Hybrid Chess Arena</CardTitle>
              <p className="mt-1 text-sm text-zinc-400">Chess.com-inspired interface with Python Backend.</p>
            </div>
            <Badge className="rounded-full bg-emerald-500/15 px-3 py-1 text-emerald-300">MVP UI</Badge>
          </CardHeader>
          <CardContent>
            <div className="rounded-3xl bg-zinc-950 p-4 border border-zinc-800 flex justify-center">
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
                  arePiecesDraggable={!engineThinking}
                  customDarkSquareStyle={{ backgroundColor: "#769656" }}
                  customLightSquareStyle={{ backgroundColor: "#eeeed2" }}
                  customBoardStyle={{ borderRadius: "8px", boxShadow: "0 10px 30px rgba(0,0,0,0.35)" }}
                />

                {showPromotionDialog && (
                  <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/50 rounded-lg">
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

        <div className="grid gap-6">
          <Card className="rounded-3xl border-zinc-800 bg-zinc-900 p-4">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-lg text-zinc-100"><Brain className="h-5 w-5" /> Game Status</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="rounded-2xl border border-zinc-800 bg-zinc-950 py-3 px-4 text-sm font-medium text-zinc-300">{status}</div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <div className="mb-2 text-sm text-zinc-400 pl-1">Play as</div>
                  <Select value={playerColor} onValueChange={(value) => newGame(value)}>
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
                  <Select value={depth} onValueChange={setDepth}>
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
              <div className="flex gap-3 pt-2">
                <Button className="flex-1 rounded-xl bg-zinc-100 text-zinc-900 hover:bg-zinc-200 py-2.5 flex justify-center items-center font-medium" onClick={() => {
                  setIsMultiplayer(false);
                  newGame(playerColor);
                }}>
                  <RotateCcw className="mr-2 h-4 w-4" /> New AI Game
                </Button>
                <Button className="flex-1 rounded-xl bg-zinc-800 text-zinc-100 hover:bg-zinc-700 py-2.5 flex justify-center items-center font-medium" onClick={maybePlayEngine} disabled={engineThinking || isMultiplayer}>
                  <Zap className="mr-2 h-4 w-4" /> AI Move
                </Button>
              </div>

              <div className="flex gap-3 pt-2 border-t border-zinc-800">
                {!isMultiplayer ? (
                  <>
                    <Button className="flex-1 rounded-xl bg-emerald-600 text-white hover:bg-emerald-500 py-2.5" onClick={startMultiplayer}>
                      <Globe className="mr-2 h-4 w-4" /> Host online
                    </Button>
                    <Button className="flex-1 rounded-xl bg-indigo-600 text-white hover:bg-indigo-500 py-2.5" onClick={joinMultiplayer}>
                      <Users className="mr-2 h-4 w-4" /> Join 
                    </Button>
                  </>
                ) : (
                  <div className="flex items-center gap-3 w-full justify-between py-1 border border-zinc-700 bg-zinc-800/50 rounded-xl px-4">
                    <span className="text-zinc-300 font-medium">Room: <span className="text-white font-mono">{roomId}</span></span>
                    <Button variant="outline" size="sm" className="h-7 bg-zinc-900 text-xs border-zinc-600" onClick={() => setIsMultiplayer(false)}>Quit</Button>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          <Card className="rounded-3xl border-zinc-800 bg-zinc-900 p-4">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-lg text-zinc-100"><Swords className="h-5 w-5" /> Move List</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="max-h-[220px] overflow-auto rounded-2xl border border-zinc-800 bg-zinc-950 custom-scrollbar">
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

          <Card className="rounded-3xl border-zinc-800 bg-zinc-900 p-4">
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

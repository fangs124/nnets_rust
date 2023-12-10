#![allow(dead_code)]
use serde::{Deserialize, Serialize};
use std::ops::{BitAnd, BitOr, BitXor, Not};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Bitboard(pub u16);
pub type BB = Bitboard;

#[allow(dead_code)]
impl Bitboard {
    pub const OUT_OF_BOUNDS: Bitboard = Bitboard(0xF888);
    pub const ONES: Bitboard = Bitboard(0x0777);
    pub const FULL_BOARD: Bitboard = Bitboard(0x0777);
    pub const EMPTY_BOARD: Bitboard = Bitboard(0x0000);
    pub const WIN_STATES: [Bitboard; 8] = [
        Bitboard(0x0700), //1st row
        Bitboard(0x0070), //2nd row
        Bitboard(0x0007), //3rd row
        Bitboard(0x0444), //1st col
        Bitboard(0x0222), //2nd col
        Bitboard(0x0111), //3rd col
        Bitboard(0x0124), //diag
        Bitboard(0x0421), //other diag
    ];
    pub const MOVES: [Bitboard; 9] = [
        Bitboard(0x0400),
        Bitboard(0x0200),
        Bitboard(0x0100),
        Bitboard(0x0040),
        Bitboard(0x0020),
        Bitboard(0x0010),
        Bitboard(0x0004),
        Bitboard(0x0002),
        Bitboard(0x0001),
    ];

    pub const POSSIBLE_MOVES: [Bitboard; 9] = BB::MOVES;

    pub fn print_full_bitboard(&self) {
        let mut s = String::new();
        for i in 0..=15 {
            if i % 4 == 3 {
                s.push('\n');
            }
            s.push(if (self.get_val() << 15) & (1 >> i) != 0 {
                '1'
            } else {
                '0'
            });
        }
        println!("{}\n", s);
    }

    pub fn print_bitboard(&self) {
        let mut s = String::new();
        for i in 0..=10 {
            if i % 4 == 3 {
                s.push('\n');
            } else {
                s.push(if (self.get_val() << 15) & (1 >> i) != 0 {
                    '1'
                } else {
                    '0'
                });
            }
        }
        println!("{}", s);
    }

    pub fn get_val(&self) -> u16 {
        self.0 //
    }
}

impl BitAnd for Bitboard {
    type Output = Bitboard;
    fn bitand(self, rhs: Bitboard) -> Self::Output {
        Bitboard(self.get_val() & rhs.get_val())
    }
}

impl BitOr for Bitboard {
    type Output = Bitboard;
    fn bitor(self, rhs: Bitboard) -> Self::Output {
        Bitboard(self.get_val() | rhs.get_val())
    }
}

impl BitXor for Bitboard {
    type Output = Bitboard;
    fn bitxor(self, rhs: Bitboard) -> Self::Output {
        Bitboard(self.get_val() ^ rhs.get_val())
    }
}

impl Not for Bitboard {
    type Output = Bitboard;
    fn not(self) -> Self::Output {
        Bitboard(!self.get_val())
    }
}

pub struct GameBoard {
    pub x_b: Bitboard,
    pub o_b: Bitboard,
    pub t_b: Bitboard,
}

pub type GB = GameBoard;

pub trait GameBoardMove {
    fn to_bitboard(&self) -> Bitboard;
}

#[derive(Debug)]
pub enum MoveError {
    OutOfBounds,
    SquareFilled,
}

#[allow(dead_code)]
#[derive(Debug, PartialEq, Eq)]
pub enum GameState {
    XWin,
    OWin,
    Tie,
    Ongoing,
}
pub type GS = GameState;

impl GameBoardMove for Bitboard {
    fn to_bitboard(&self) -> Bitboard {
        *self
    }
}

impl GameBoardMove for [u16; 2] {
    fn to_bitboard(&self) -> Bitboard {
        if !self[0] <= 2 || !self[1] <= 2 {
            panic!("to_bitboard: out of bounds!");
        }
        BB::POSSIBLE_MOVES[usize::from(self[0] + 3 * self[1])]
    }
}

#[allow(dead_code)]
impl GameBoard {
    pub const NEW_GAMEBOARD: Self = Self {
        x_b: BB::EMPTY_BOARD,
        o_b: BB::EMPTY_BOARD,
        t_b: BB::EMPTY_BOARD,
    };

    pub fn new_game(&mut self) {
        self.x_b = BB::EMPTY_BOARD;
        self.o_b = BB::EMPTY_BOARD;
        self.t_b = BB::EMPTY_BOARD;
    }

    pub fn print_gameboard(&self) {
        let mut s = String::new();
        for i in 0..11 {
            s.push_str(if i % 4 == 3 {
                "\n"
            } else if self.o_b & Bitboard((1 << 10) >> i) != BB::EMPTY_BOARD {
                "O"
            } else if self.x_b & Bitboard((1 << 10) >> i) != BB::EMPTY_BOARD {
                "X"
            } else {
                "."
            })
        }
        println!("==== GAMEBOARD ====");
        println!("{}", s);
        println!("===================");
    }

    pub fn valid_moves(&self) -> Vec<Bitboard> {
        let mut mvs = Vec::new();
        for m in BB::POSSIBLE_MOVES {
            if self.is_valid_move(&m) {
                mvs.push(m)
            };
        }
        mvs
    }

    pub fn game_state(&self) -> GameState {
        for b in BB::WIN_STATES {
            if self.x_b & b == b {
                return GameState::XWin;
            } else if self.o_b & b == b {
                return GameState::OWin;
            }
        }

        if (self.x_b | self.o_b) == BB::FULL_BOARD {
            GameState::Tie
        } else {
            GameState::Ongoing
        }
    }

    pub fn make_move<T: GameBoardMove>(&mut self, m: T) -> Result<(), MoveError> {
        if m.to_bitboard() & BB::OUT_OF_BOUNDS != BB::EMPTY_BOARD {
            return Err(MoveError::OutOfBounds);
        } else if m.to_bitboard() & (self.x_b | self.o_b) != BB::EMPTY_BOARD {
            return Err(MoveError::SquareFilled);
        }

        match self.t_b & BB::FULL_BOARD {
            BB::EMPTY_BOARD => {
                self.x_b = self.x_b | m.to_bitboard();
                self.t_b = !self.t_b;
            }
            _ => {
                self.o_b = self.o_b | m.to_bitboard();
                self.t_b = !self.t_b;
            }
        }
        return Ok(());
    }

    pub fn is_valid_move<T: GameBoardMove>(&self, m: &T) -> bool {
        return m.to_bitboard() & (self.x_b | self.o_b) == BB::EMPTY_BOARD;
    }
}

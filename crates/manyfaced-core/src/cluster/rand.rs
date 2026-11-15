//! dlib's default random number generator (`dlib::rand`), needed to reproduce
//! `chinese_whispers` labels exactly.
//!
//! Ported from `dlib/rand/rand_kernel_1.h` and `dlib/rand/mersenne_twister.h` (dlib v20.0.1,
//! Boost Software License 1.0, see `THIRD_PARTY_NOTICES`): a standard MT19937 seeded with 5489,
//! advanced by 10 000 draws on construction.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// MT19937 (`dlib::mt19937`, equal to `std::mt19937`).
#[derive(Clone)]
pub struct Mt19937 {
    state: [u32; N],
    index: usize,
}

impl Mt19937 {
    pub fn new(seed: u32) -> Self {
        let mut state = [0u32; N];
        state[0] = seed;
        for i in 1..N {
            let prev = state[i - 1];
            state[i] = 1_812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        Mt19937 { state, index: N }
    }

    fn twist(&mut self) {
        for i in 0..N {
            let y = (self.state[i] & UPPER_MASK) | (self.state[(i + 1) % N] & LOWER_MASK);
            let mut next = self.state[(i + M) % N] ^ (y >> 1);
            if y & 1 != 0 {
                next ^= MATRIX_A;
            }
            self.state[i] = next;
        }
        self.index = 0;
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            self.twist();
        }
        let mut z = self.state[self.index];
        self.index += 1;
        z ^= z >> 11;
        z ^= (z << 7) & 0x9d2c_5680;
        z ^= (z << 15) & 0xefc6_0000;
        z ^ (z >> 18)
    }
}

/// `dlib::rand` in its default-constructed state.
#[derive(Clone)]
pub struct DlibRand {
    mt: Mt19937,
}

impl Default for DlibRand {
    fn default() -> Self {
        let mut mt = Mt19937::new(5489);
        // `rand::init()`: "prime the generator a bit".
        for _ in 0..10_000 {
            mt.next_u32();
        }
        DlibRand { mt }
    }
}

impl DlibRand {
    pub fn get_random_32bit_number(&mut self) -> u32 {
        self.mt.next_u32()
    }

    /// `(a << 32) | b` with `a` drawn first.
    pub fn get_random_64bit_number(&mut self) -> u64 {
        let a = u64::from(self.get_random_32bit_number());
        let b = u64::from(self.get_random_32bit_number());
        (a << 32) | b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_matches_the_cpp_standard() {
        // [rand.predef]: the 10000th consecutive invocation of a default-constructed
        // std::mt19937 produces 4123659995.
        let mut mt = Mt19937::new(5489);
        let mut last = 0;
        for _ in 0..10_000 {
            last = mt.next_u32();
        }
        assert_eq!(last, 4_123_659_995);
        assert_eq!(Mt19937::new(5489).next_u32(), 3_499_211_612);
    }
}

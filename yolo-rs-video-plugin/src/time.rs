extern crate ffmpeg_next;

use ffmpeg_next::util::mathematics::rescale::{Rescale, TIME_BASE};
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Time {
    time_base: ffmpeg_next::Rational,
    time: Option<i64>,
}

impl Time {
    pub fn from_secs_f64(secs: f64) -> Self {
        Self {
            time: Some((secs * TIME_BASE.denominator() as f64).round() as i64),
            time_base: TIME_BASE,
        }
    }

    pub fn zero() -> Self {
        Time {
            time: Some(0),
            time_base: (1, 90000).into(),
        }
    }

    pub fn aligned_with(&self, rhs: &Time) -> Aligned {
        Aligned {
            lhs: self.time,
            rhs: rhs
                .time
                .map(|rhs_time| rhs_time.rescale(rhs.time_base, self.time_base)),
            time_base: self.time_base,
        }
    }

    pub fn into_value(self) -> Option<i64> {
        self.time
    }

    pub(crate) fn aligned_with_rational(&self, time_base: ffmpeg_next::Rational) -> Time {
        Time {
            time: self
                .time
                .map(|time| time.rescale(self.time_base, time_base)),
            time_base,
        }
    }
}

impl From<Duration> for Time {
    #[inline]
    fn from(duration: Duration) -> Self {
        Time::from_secs_f64(duration.as_secs_f64())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Aligned {
    lhs: Option<i64>,
    rhs: Option<i64>,
    time_base: ffmpeg_next::Rational,
}

impl Aligned {
    /// Add two timestamps together.
    pub fn add(self) -> Time {
        match (self.lhs, self.rhs) {
            (Some(lhs_time), Some(rhs_time)) => Time {
                time: Some(lhs_time + rhs_time),
                time_base: self.time_base,
            },
            _ => Time {
                time: None,
                time_base: self.time_base,
            },
        }
    }
}

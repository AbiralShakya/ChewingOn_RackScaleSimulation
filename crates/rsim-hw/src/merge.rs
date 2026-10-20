//! Deep merge of YAML values: lets a user start from a preset and override
//! any nested field (`overrides: { l2: { hit_latency: 280 } }`).

use serde_yaml::Value;

pub fn deep_merge(base: &mut Value, over: Value) {
    match (base, over) {
        (Value::Mapping(b), Value::Mapping(o)) => {
            for (k, v) in o {
                match b.get_mut(&k) {
                    Some(slot) => deep_merge(slot, v),
                    None => {
                        b.insert(k, v);
                    }
                }
            }
        }
        (slot, v) => *slot = v,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nested_override() {
        let mut a: Value = serde_yaml::from_str("a: {b: 1, c: 2}\nd: 3").unwrap();
        let b: Value = serde_yaml::from_str("a: {c: 5}").unwrap();
        deep_merge(&mut a, b);
        let want: Value = serde_yaml::from_str("a: {b: 1, c: 5}\nd: 3").unwrap();
        assert_eq!(a, want);
    }
}

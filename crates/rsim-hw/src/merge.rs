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

/// Keys the user wrote that did not survive deserialization. serde drops
/// unknown fields silently, so a typo such as `sm_clok_mhz` in an override
/// would otherwise leave the preset unchanged without a word. `resolved` is
/// the parsed value serialized back; every key path in `user` must exist in
/// it. Returns one message per unknown key, with the closest known sibling
/// (or the closest of `optional`: field names that may be absent from
/// `resolved` because they are skipped when unset).
pub fn unknown_keys(user: &Value, resolved: &Value, optional: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    walk(user, resolved, "", optional, &mut out);
    out
}

fn walk(user: &Value, resolved: &Value, path: &str, optional: &[&str], out: &mut Vec<String>) {
    match (user, resolved) {
        (Value::Mapping(u), Value::Mapping(r)) => {
            for (k, v) in u {
                let Some(name) = k.as_str() else { continue };
                let here = if path.is_empty() {
                    name.to_string()
                } else {
                    format!("{path}.{name}")
                };
                match r.get(k) {
                    Some(rv) => walk(v, rv, &here, optional, out),
                    // Explicit nulls mean "unset"; optional fields that stay
                    // unset are skipped when serialized.
                    None if v.is_null() => {}
                    None => {
                        let mut known: Vec<&str> = r.keys().filter_map(Value::as_str).collect();
                        known.extend_from_slice(optional);
                        let hint = closest(name, &known)
                            .map(|c| format!(" (did you mean `{c}`?)"))
                            .unwrap_or_default();
                        out.push(format!("unknown key `{here}`{hint}"));
                    }
                }
            }
        }
        (Value::Sequence(u), Value::Sequence(r)) => {
            for (i, (uv, rv)) in u.iter().zip(r).enumerate() {
                walk(uv, rv, &format!("{path}[{i}]"), optional, out);
            }
        }
        _ => {}
    }
}

/// The known key nearest to `name` by edit distance, if it is close enough
/// to be a plausible typo.
fn closest<'a>(name: &str, known: &[&'a str]) -> Option<&'a str> {
    known
        .iter()
        .map(|k| (edit_distance(name, k), *k))
        .filter(|&(d, k)| d <= 2.max(k.len() / 3))
        .min()
        .map(|(_, k)| k)
}

fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, &cb) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + (ca != cb) as usize)
                .min(prev[j + 1] + 1)
                .min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_key_with_suggestion() {
        let user: Value =
            serde_yaml::from_str("a: {sm_clok_mhz: 1, b: 2}\nxs: [{c: 1}, {d: 2}]").unwrap();
        let res: Value =
            serde_yaml::from_str("a: {sm_clock_mhz: 1, b: 2}\nxs: [{c: 1}, {c: 2}]").unwrap();
        let errs = unknown_keys(&user, &res, &[]);
        assert_eq!(errs.len(), 2, "{errs:?}");
        assert!(errs[0].contains("`a.sm_clok_mhz` (did you mean `sm_clock_mhz`?)"));
        assert!(errs[1].contains("`xs[1].d`"));
    }
    #[test]
    fn nested_override() {
        let mut a: Value = serde_yaml::from_str("a: {b: 1, c: 2}\nd: 3").unwrap();
        let b: Value = serde_yaml::from_str("a: {c: 5}").unwrap();
        deep_merge(&mut a, b);
        let want: Value = serde_yaml::from_str("a: {b: 1, c: 5}\nd: 3").unwrap();
        assert_eq!(a, want);
    }
}

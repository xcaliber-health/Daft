//! JSON trees of any depth: serialized and dropped without overflowing a stack.
//!
//! A plan rendered as JSON nests one object per plan level. Serializing a
//! nested value recurses once per level, and so does dropping it. Serializing
//! here grows the stack whenever it runs low, and dropping takes the tree apart
//! one level at a time, so a plan of any depth renders on any thread.

use serde::{
    Serialize, Serializer,
    ser::{SerializeMap, SerializeSeq},
};
use serde_json::Value;

/// Serializes `value` as compact JSON, exactly as [`serde_json::to_string`]
/// does, at any nesting depth.
///
/// # Errors
/// Returns an error if a map key or number cannot be written as JSON.
pub fn to_string(value: &Value) -> serde_json::Result<String> {
    serde_json::to_string(&Growing(value))
}

/// Drops `value` one level at a time, at any nesting depth.
///
/// A nested value's own drop recurses once per level; this moves each level's
/// children onto a heap stack before dropping it.
pub fn dismantle(value: Value) {
    let mut pending = vec![value];
    while let Some(value) = pending.pop() {
        match value {
            Value::Array(items) => pending.extend(items),
            Value::Object(map) => pending.extend(map.into_iter().map(|(_, child)| child)),
            _ => {}
        }
    }
}

/// A value serialized on a stack grown when the current one runs low.
struct Growing<'a>(&'a Value);

impl Serialize for Growing<'_> {
    #[recursive::recursive]
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            Value::Array(items) => {
                let mut seq = serializer.serialize_seq(Some(items.len()))?;
                for item in items {
                    seq.serialize_element(&Growing(item))?;
                }
                seq.end()
            }
            Value::Object(map) => {
                let mut entries = serializer.serialize_map(Some(map.len()))?;
                for (key, child) in map {
                    entries.serialize_entry(key, &Growing(child))?;
                }
                entries.end()
            }
            leaf => leaf.serialize(serializer),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// A plan-shaped tree `depth` levels deep, each level moved into the next.
    fn nested(depth: usize) -> Value {
        (0..depth).fold(json!({"name": "leaf", "children": []}), |child, level| {
            let mut node = serde_json::Map::new();
            node.insert("name".into(), Value::String(format!("node {level}")));
            node.insert("id".into(), Value::from(level));
            node.insert("children".into(), Value::Array(vec![child]));
            Value::Object(node)
        })
    }

    #[test]
    fn it_serializes_as_serde_json_does() {
        let value =
            json!({"b": [1, 2.5, null, true, "x"], "a": {"nested": {"deeper": []}}, "c": ""});

        assert_eq!(
            to_string(&value).unwrap(),
            serde_json::to_string(&value).unwrap()
        );
    }

    #[test]
    fn a_tree_deeper_than_its_threads_stack_serializes_and_drops() {
        // Far more levels than a 256 KiB stack holds unguarded.
        let depth = 50_000;
        let length = std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(move || {
                let value = nested(depth);
                let rendered = to_string(&value).unwrap();
                dismantle(value);
                rendered.len()
            })
            .unwrap()
            .join()
            .unwrap();

        assert!(length > depth * "{\"children\":[".len());
    }
}

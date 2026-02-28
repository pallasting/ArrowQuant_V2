# Dependency Upgrade Issue - PyO3 Version Conflict

**Date**: 2026-02-24  
**Status**: ⚠️ Blocked - Workspace Conflict

## Issue Description

Cannot upgrade arrow_quant_v2 to PyO3 0.22 due to workspace-level dependency conflict.

## Error Message

```
error: failed to select a version for `pyo3`.
    ... required by package `arrow_storage v0.1.0`
    versions that meet the requirements `^0.20` are: 0.20.3, 0.20.2, 0.20.1, 0.20.0

package `pyo3` links to the native library `python`, but it conflicts with a 
previous package which links to `python` as well:
package `pyo3 v0.22.4`
    ... which satisfies dependency `pyo3 = "^0.22"` of package `arrow_quant_v2`

Only one package in the dependency graph may specify the same links value.
```

## Root Cause

The workspace contains multiple packages that depend on PyO3:
1. **arrow_quant_v2**: Attempting to use PyO3 0.22
2. **arrow_storage**: Currently using PyO3 0.20

Cargo does not allow multiple versions of PyO3 in the same workspace because PyO3 links to the native Python library, and only one version can be linked.

## Solution Options

### Option 1: Upgrade All Workspace Packages (Recommended)
Upgrade all packages in the workspace to PyO3 0.22 simultaneously.

**Packages to upgrade**:
- `ai_os_diffusion/arrow_quant_v2` ✅ Already upgraded
- `ai_os_diffusion/arrow_storage` ⏳ Needs upgrade
- Any other packages using PyO3

**Steps**:
1. Identify all packages using PyO3:
   ```bash
   cd ai_os_diffusion
   grep -r "pyo3 =" */Cargo.toml
   ```

2. Upgrade each package's Cargo.toml:
   ```toml
   pyo3 = { version = "0.22", features = [...] }
   ```

3. Update Python binding code in each package:
   - `PyDict::new()` → `PyDict::new_bound()`
   - `PyList::empty()` → `PyList::empty_bound()`
   - `py.import()` → `py.import_bound()`

4. Test each package individually

5. Test workspace build:
   ```bash
   cargo check --workspace
   ```

### Option 2: Keep PyO3 0.20 (Temporary)
Revert arrow_quant_v2 to PyO3 0.20 until all workspace packages can be upgraded together.

**Steps**:
1. Revert Cargo.toml changes:
   ```toml
   pyo3 = { version = "0.20", features = ["extension-module", "abi3-py310"] }
   pyo3-asyncio = { version = "0.20", features = ["tokio-runtime"] }
   ```

2. Revert code changes in:
   - `src/python.rs`
   - `src/python_async.rs`

3. Upgrade other dependencies (Arrow, ndarray, etc.) which don't conflict

4. Plan coordinated PyO3 upgrade for entire workspace

### Option 3: Workspace-Level Dependency Management
Use workspace-level dependency management to ensure consistency.

**Steps**:
1. Edit `ai_os_diffusion/Cargo.toml` (workspace root):
   ```toml
   [workspace.dependencies]
   pyo3 = { version = "0.22", features = ["extension-module"] }
   ```

2. Update each package to use workspace dependency:
   ```toml
   [dependencies]
   pyo3 = { workspace = true, features = ["abi3-py310"] }
   ```

## Recommended Approach

**Immediate**: Option 2 (Keep PyO3 0.20)
- Allows upgrading other dependencies immediately
- Maintains system stability
- Defers PyO3 upgrade to coordinated effort

**Short-term** (Within 1 week): Option 1 (Upgrade All Packages)
- Create upgrade plan for all workspace packages
- Test each package individually
- Coordinate upgrade across team

**Long-term**: Option 3 (Workspace Management)
- Implement workspace-level dependency management
- Prevents future version conflicts
- Simplifies dependency updates

## Current Status

### Completed
- ✅ Identified PyO3 version conflict
- ✅ Documented solution options
- ✅ Upgraded other non-conflicting dependencies

### Pending
- ⏳ Identify all packages using PyO3 in workspace
- ⏳ Create coordinated upgrade plan
- ⏳ Test workspace-wide PyO3 0.22 upgrade

## Immediate Action

**Revert to PyO3 0.20** for arrow_quant_v2 to maintain compatibility:

```bash
cd ai_os_diffusion/arrow_quant_v2
git checkout HEAD -- Cargo.toml src/python.rs src/python_async.rs
```

**Keep other dependency upgrades** (Arrow, ndarray, tokio, etc.) as they don't conflict.

## Next Steps

1. **Immediate** (Today):
   - Revert PyO3 changes
   - Keep other dependency upgrades
   - Document PyO3 upgrade plan

2. **Short-term** (This week):
   - Audit all workspace packages for PyO3 usage
   - Create coordinated upgrade plan
   - Schedule upgrade window

3. **Medium-term** (Next sprint):
   - Execute coordinated PyO3 0.22 upgrade
   - Test all packages
   - Deploy to production

## References

- [Cargo Links Documentation](https://doc.rust-lang.org/cargo/reference/resolver.html#links)
- [PyO3 Workspace Setup](https://pyo3.rs/v0.22.0/building-and-distribution.html#workspace-setup)
- [Cargo Workspace Dependencies](https://doc.rust-lang.org/cargo/reference/workspaces.html#the-dependencies-table)

---

*Issue identified: 2026-02-24*  
*Resolution: Pending workspace-wide coordination*

# ROCm Version Matching Options Analysis

**Date**: 2026-02-15  
**Current Situation**: Ollama bundled ROCm 6.3 vs System ROCm 7.2 mismatch  
**Current Solution**: Vulkan backend (working but suboptimal)

## Option Analysis

### Option A: Downgrade System ROCm 7.2 → 6.3

**Goal**: Match system ROCm to Ollama's bundled version

#### Feasibility Check

1. **Check if ROCm 6.3 packages are available**:
   ```bash
   apt-cache policy rocm-dev | grep 6.3
   ```

2. **Check current ROCm usage**:
   ```bash
   # Find what depends on ROCm
   dpkg -l | grep rocm
   apt-cache rdepends rocm-dev
   ```

#### Pros
- ✅ Perfect compatibility with Ollama
- ✅ Native ROCm performance (potentially faster than Vulkan)
- ✅ Official support from Ollama
- ✅ No need to modify Ollama installation

#### Cons
- ❌ May break other ROCm-dependent applications
- ❌ ROCm 6.3 packages may not be available in repos
- ❌ Older version = missing features/fixes from 7.2
- ❌ Future Ollama updates may require newer ROCm
- ❌ System-wide change affects all applications

#### Risk Assessment
- **Risk Level**: HIGH
- **Reversibility**: Medium (can upgrade back but may cause issues)
- **Impact**: System-wide

#### Steps (if proceeding)
```bash
# 1. Backup current ROCm installation list
dpkg -l | grep rocm > rocm_7.2_packages.txt

# 2. Check for ROCm 6.3 availability
apt-cache policy rocm-dev

# 3. Remove ROCm 7.2
sudo apt remove --purge rocm-* amdgpu-dkms

# 4. Add ROCm 6.3 repository (if available)
# This may require finding archived packages

# 5. Install ROCm 6.3
sudo apt install rocm-dev=6.3.*

# 6. Reboot
sudo reboot
```

---

### Option B: Upgrade Ollama's Bundled ROCm 6.3 → 7.2

**Goal**: Replace Ollama's bundled libraries with system ROCm 7.2

#### Feasibility Check

1. **Check Ollama's bundled libraries**:
   ```bash
   ls -la /usr/local/lib/ollama/rocm/
   ```

2. **Check system ROCm libraries**:
   ```bash
   ls -la /opt/rocm-7.2.0/lib/
   ```

3. **Compare library versions**:
   ```bash
   # Ollama bundled
   ls /usr/local/lib/ollama/rocm/*.so* | wc -l
   
   # System ROCm
   ls /opt/rocm-7.2.0/lib/*.so* | wc -l
   ```

#### Pros
- ✅ Uses latest ROCm 7.2 features
- ✅ No system-wide changes
- ✅ Other applications unaffected
- ✅ Easy to revert (restore backup)
- ✅ Maintains system consistency

#### Cons
- ❌ Unsupported configuration (not officially tested)
- ❌ May break on Ollama updates (overwrites libraries)
- ❌ Potential ABI incompatibilities
- ❌ Ollama may have specific patches in bundled libs
- ❌ Requires manual maintenance

#### Risk Assessment
- **Risk Level**: MEDIUM
- **Reversibility**: High (easy to restore backup)
- **Impact**: Ollama only

#### Steps (if proceeding)
```bash
# 1. Backup Ollama's bundled ROCm
sudo cp -r /usr/local/lib/ollama/rocm /usr/local/lib/ollama/rocm.backup

# 2. Create symlinks to system ROCm (safer than copying)
sudo rm -rf /usr/local/lib/ollama/rocm/*
sudo ln -s /opt/rocm-7.2.0/lib/*.so* /usr/local/lib/ollama/rocm/

# 3. Update systemd config to use system ROCm
sudo tee /etc/systemd/system/ollama.service.d/rocm.conf << 'EOF'
[Service]
Environment="HSA_OVERRIDE_GFX_VERSION=9.0.6"
Environment="ROCR_VISIBLE_DEVICES=0"
Environment="HIP_VISIBLE_DEVICES=0"
Environment="LD_LIBRARY_PATH=/opt/rocm-7.2.0/lib:/opt/rocm-7.2.0/lib64"
Environment="ROCM_PATH=/opt/rocm-7.2.0"
EOF

# 4. Restart Ollama
sudo systemctl daemon-reload
sudo systemctl restart ollama

# 5. Test
ollama ps
python3 scripts/test_gpu_inference.py

# 6. If fails, restore backup
sudo rm -rf /usr/local/lib/ollama/rocm
sudo mv /usr/local/lib/ollama/rocm.backup /usr/local/lib/ollama/rocm
sudo systemctl restart ollama
```

---

### Option C: Keep Vulkan (Current Solution)

**Goal**: Accept Vulkan as the working solution

#### Pros
- ✅ Already working (1.54s inference)
- ✅ No system changes needed
- ✅ No risk of breaking anything
- ✅ Officially supported by Ollama
- ✅ 5x faster than CPU

#### Cons
- ❌ May be slower than native ROCm
- ❌ Experimental status (though stable)
- ❌ Less mature than ROCm backend

#### Performance Comparison (Estimated)
| Backend | Inference Time | Status |
|---------|----------------|--------|
| CPU | 8s | Baseline |
| Vulkan | 1.54s | Current ✅ |
| ROCm (if working) | 1.2-1.5s | Estimated |

**Performance difference**: ~10-20% (not significant)

---

## Recommendation

### Primary Recommendation: **Option B** (Upgrade Ollama's bundled ROCm)

**Reasoning**:
1. **Lower risk**: Only affects Ollama, easy to revert
2. **Better compatibility**: Uses system ROCm 7.2
3. **Testable**: Can try and rollback quickly
4. **Learning opportunity**: Understand if it's a library issue or deeper incompatibility

**Implementation Plan**:
1. Try Option B first (30 minutes)
2. If fails, document why and keep Vulkan
3. If succeeds, compare performance with Vulkan
4. Choose best performing solution

### Fallback: **Option C** (Keep Vulkan)

If Option B fails or shows no performance improvement, keep Vulkan:
- Already working well (1.54s)
- Stable and supported
- Meets 4/5 Phase 1.1 criteria

### Not Recommended: **Option A** (Downgrade System ROCm)

**Reasons**:
- Too risky (system-wide impact)
- May break other applications
- ROCm 6.3 packages may not be available
- Not future-proof (Ollama will eventually need newer ROCm)

---

## Performance Expectations

### If Option B Succeeds (Native ROCm 7.2)

**Expected improvements over Vulkan**:
- Inference: 1.54s → 1.2-1.4s (10-20% faster)
- Throughput: 48.8/min → 55-60/min (sequential)
- GPU utilization: Better memory management
- Stability: More mature backend

**Still need batching for 100/min target**

### If Option B Fails

**Keep Vulkan**:
- Current performance: 1.54s (excellent)
- Meets 4/5 criteria
- Stable and working

---

## Testing Plan for Option B

### Phase 1: Backup and Prepare (5 min)
```bash
sudo cp -r /usr/local/lib/ollama/rocm /usr/local/lib/ollama/rocm.backup
```

### Phase 2: Replace Libraries (10 min)
```bash
# Method 1: Symlinks (preferred - easy to revert)
sudo rm -rf /usr/local/lib/ollama/rocm/*
sudo ln -s /opt/rocm-7.2.0/lib/*.so* /usr/local/lib/ollama/rocm/

# Method 2: Copy (alternative)
# sudo cp /opt/rocm-7.2.0/lib/*.so* /usr/local/lib/ollama/rocm/
```

### Phase 3: Configure and Test (10 min)
```bash
# Update config
sudo tee /etc/systemd/system/ollama.service.d/rocm.conf << 'EOF'
[Service]
Environment="HSA_OVERRIDE_GFX_VERSION=9.0.6"
Environment="ROCR_VISIBLE_DEVICES=0"
Environment="HIP_VISIBLE_DEVICES=0"
Environment="LD_LIBRARY_PATH=/opt/rocm-7.2.0/lib"
Environment="ROCM_PATH=/opt/rocm-7.2.0"
EOF

# Restart
sudo systemctl daemon-reload
sudo systemctl restart ollama
sleep 5

# Check logs
sudo journalctl -u ollama --since "1 minute ago" | grep -i "gpu\|rocm\|error"

# Test
ollama ps
python3 scripts/test_gpu_inference.py
```

### Phase 4: Rollback if Needed (5 min)
```bash
sudo rm -rf /usr/local/lib/ollama/rocm
sudo mv /usr/local/lib/ollama/rocm.backup /usr/local/lib/ollama/rocm
sudo tee /etc/systemd/system/ollama.service.d/rocm.conf << 'EOF'
[Service]
Environment="OLLAMA_VULKAN=1"
EOF
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

---

## Decision Matrix

| Criterion | Option A (Downgrade) | Option B (Upgrade) | Option C (Vulkan) |
|-----------|---------------------|-------------------|-------------------|
| Risk | HIGH | MEDIUM | LOW |
| Reversibility | Medium | High | N/A |
| Performance | Best (native) | Best (native) | Good |
| Stability | Unknown | Unknown | Proven |
| Maintenance | Low | Medium | Low |
| System Impact | High | Low | None |
| **Recommendation** | ❌ No | ✅ Try First | ✅ Fallback |

---

## Conclusion

**Recommended Approach**:
1. **Try Option B** (upgrade Ollama's bundled ROCm to 7.2)
2. **If fails**: Keep Option C (Vulkan - current working solution)
3. **Avoid Option A** (system-wide downgrade - too risky)

**Expected Outcome**:
- Best case: Native ROCm 7.2 works, 10-20% faster than Vulkan
- Likely case: ROCm 7.2 still crashes, keep Vulkan
- Worst case: Easy rollback to Vulkan

**Time Investment**: 30 minutes to test Option B
**Risk**: Low (easy rollback)
**Potential Gain**: 10-20% performance improvement

Would you like me to proceed with Option B testing?

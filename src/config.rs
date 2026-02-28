//! Configuration types for ArrowQuant V2

use crate::errors::{QuantError, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Modality of the diffusion model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    Text,
    Code,
    Image,
    Audio,
}

impl std::fmt::Display for Modality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Modality::Text => write!(f, "text"),
            Modality::Code => write!(f, "code"),
            Modality::Image => write!(f, "image"),
            Modality::Audio => write!(f, "audio"),
        }
    }
}

/// Quantization method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantMethod {
    /// Residual Refinement Quantization (for INT2)
    R2Q,
    /// Generalized Post-Training Quantization (Hessian-based)
    GPTQ,
    /// Simple Post-Training Quantization (MinMax)
    PTQ,
}

/// Quantization strategy combining method and enhancements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizationStrategy {
    pub method: QuantMethod,
    pub time_aware: bool,
    pub spatial: bool,
}

/// Configuration for diffusion model quantization
///
/// This is the primary configuration struct for ArrowQuant V2 diffusion quantization.
/// It supports deployment profiles (edge/local/cloud) and provides validation.
///
/// # Examples
///
/// ```
/// use arrow_quant_v2::{DiffusionQuantConfig, DeploymentProfile};
///
/// // Create from deployment profile
/// let config = DiffusionQuantConfig::from_profile(DeploymentProfile::Edge);
///
/// // Validate configuration
/// config.validate().expect("Invalid configuration");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionQuantConfig {
    /// Target bit width (2, 4, or 8)
    pub bit_width: u8,

    /// Modality (auto-detected if None)
    pub modality: Option<Modality>,

    /// Number of time groups for time-aware quantization
    pub num_time_groups: usize,

    /// Group size for per-group quantization
    pub group_size: usize,

    /// Enable time-aware quantization
    pub enable_time_aware: bool,

    /// Enable spatial quantization
    pub enable_spatial: bool,

    /// Minimum accuracy threshold (cosine similarity)
    pub min_accuracy: f32,

    /// Number of calibration samples
    pub calibration_samples: usize,

    /// Deployment profile
    pub deployment_profile: DeploymentProfile,

    /// Enable fail-fast mode (disable fallback on quantization failure)
    #[serde(default)]
    pub fail_fast: bool,

    /// Number of parallel threads for layer quantization (0 = auto-detect)
    #[serde(default)]
    pub num_threads: usize,

    /// Enable streaming mode (load one layer at a time to minimize memory)
    #[serde(default)]
    pub enable_streaming: bool,

    /// Skip quantization for sensitive layers (preserve FP16)
    #[serde(default)]
    pub skip_sensitive_layers: bool,

    /// User-defined list of sensitive layer names (exact match)
    #[serde(default)]
    pub sensitive_layer_names: Vec<String>,

    /// Regex patterns for sensitive layer detection
    #[serde(default)]
    pub sensitive_layer_patterns: Vec<String>,

    /// Enable per-layer bit-width selection (mixed-precision quantization)
    #[serde(default)]
    pub enable_mixed_precision: bool,

    /// Enable thermodynamic entropy-based dynamic bit-width selection
    #[serde(default)]
    pub enable_entropy_adaptation: bool,

    /// Per-layer bit-width assignments (layer_name -> bit_width)
    /// If a layer is not in this map, uses the default bit_width
    /// Supported values: 2, 4, 8, 16 (16 = FP16, no quantization)
    #[serde(default)]
    pub layer_bit_widths: std::collections::HashMap<String, u8>,

    /// Target model size in MB for automatic bit-width optimization
    /// If set, the system will automatically assign bit-widths to meet this target
    #[serde(default)]
    pub target_model_size_mb: Option<f32>,

    /// Enable memory-aware scheduling (pipelined quantization with memory safety)
    #[serde(default)]
    pub enable_memory_aware_scheduling: bool,

    /// Maximum memory limit in MB for pipelined quantization (None = auto-detect)
    #[serde(default)]
    pub max_memory_limit_mb: Option<usize>,

    /// Thermodynamic constraint configuration (Phase 1: Validation)
    #[serde(default)]
    pub thermodynamic: ThermodynamicConfig,
}

/// Deployment profile presets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeploymentProfile {
    Edge,
    Local,
    Cloud,
}

/// Interpolation method for boundary smoothing (Phase 2)
///
/// Different methods provide different trade-offs between smoothness,
/// computational cost, and parameter preservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InterpolationMethod {
    /// Linear interpolation between boundary parameters
    ///
    /// Formula: `value(t) = left * (1 - α) + right * α`
    /// where `α = (t - t_left) / (t_right - t_left)`
    ///
    /// - Pros: Fast, simple, predictable
    /// - Cons: C⁰ continuity only (discontinuous derivatives)
    Linear,

    /// Cubic spline interpolation for smooth transitions
    ///
    /// Uses cubic polynomials to ensure C² continuity (continuous second derivatives).
    /// Provides smoother transitions than linear interpolation.
    ///
    /// - Pros: C² continuity, very smooth
    /// - Cons: More complex, slightly slower
    Cubic,

    /// Sigmoid-based interpolation for gradual transitions
    ///
    /// Formula: `α_sigmoid = 1 / (1 + exp(-k*(α - 0.5)))`
    /// where k controls steepness (default: 10)
    ///
    /// - Pros: Gradual transitions, natural-looking curves
    /// - Cons: Non-linear, may overshoot slightly
    Sigmoid,
}

impl std::fmt::Display for InterpolationMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InterpolationMethod::Linear => write!(f, "linear"),
            InterpolationMethod::Cubic => write!(f, "cubic"),
            InterpolationMethod::Sigmoid => write!(f, "sigmoid"),
        }
    }
}

impl From<InterpolationMethod> for crate::thermodynamic::InterpolationMethod {
    fn from(method: InterpolationMethod) -> Self {
        match method {
            InterpolationMethod::Linear => crate::thermodynamic::InterpolationMethod::Linear,
            InterpolationMethod::Cubic => crate::thermodynamic::InterpolationMethod::Cubic,
            InterpolationMethod::Sigmoid => crate::thermodynamic::InterpolationMethod::Sigmoid,
        }
    }
}

/// Configuration for Markov validation (Phase 1)
///
/// Validates smoothness of quantization parameters across time groups
/// to detect violations of the Markov property in diffusion models.
///
/// # Examples
///
/// ```
/// use arrow_quant_v2::ValidationConfig;
///
/// let config = ValidationConfig {
///     enabled: true,
///     smoothness_threshold: 0.3,
///     log_violations: true,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Enable Markov validation
    #[serde(default = "default_validation_enabled")]
    pub enabled: bool,

    /// Smoothness threshold for violation detection (default: 0.3 = 30% jump)
    /// Jumps exceeding this threshold are logged as violations
    #[serde(default = "default_smoothness_threshold")]
    pub smoothness_threshold: f32,

    /// Log violations at WARN level
    #[serde(default = "default_log_violations")]
    pub log_violations: bool,
}

fn default_validation_enabled() -> bool {
    cfg!(debug_assertions) // Enabled in debug, disabled in release
}

fn default_smoothness_threshold() -> f32 {
    0.3 // 30% jump threshold
}

fn default_log_violations() -> bool {
    true
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enabled: default_validation_enabled(),
            smoothness_threshold: default_smoothness_threshold(),
            log_violations: default_log_violations(),
        }
    }
}

impl ValidationConfig {
    /// Validate configuration values
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `smoothness_threshold` is not in range [0.0, 1.0]
    ///
    /// # Examples
    ///
    /// ```
    /// use arrow_quant_v2::ValidationConfig;
    ///
    /// let config = ValidationConfig::default();
    /// config.validate().expect("Invalid configuration");
    /// ```
    pub fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.smoothness_threshold) {
            return Err(QuantError::ConfigurationError(format!(
                "smoothness_threshold must be in range [0.0, 1.0], got {}",
                self.smoothness_threshold
            )));
        }
        Ok(())
    }
}

/// Configuration for boundary smoothing (Phase 2)
///
/// Smooths parameter transitions at time group boundaries to reduce parameter jumps
/// and improve Markov chain properties. Supports configurable window size and
/// multiple interpolation methods.
///
/// # Examples
///
/// ```
/// use arrow_quant_v2::{BoundarySmoothingConfig, InterpolationMethod};
///
/// let config = BoundarySmoothingConfig {
///     enabled: true,
///     window_size: 5,
///     interpolation: InterpolationMethod::Linear,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundarySmoothingConfig {
    /// Enable boundary smoothing
    #[serde(default = "default_smoothing_enabled")]
    pub enabled: bool,

    /// Smoothing window size (number of timesteps on each side of boundary)
    /// Range: 1-20, default: 5
    #[serde(default = "default_window_size")]
    pub window_size: usize,

    /// Interpolation method for smoothing
    #[serde(default = "default_interpolation_method")]
    pub interpolation: InterpolationMethod,
}

fn default_smoothing_enabled() -> bool {
    false // Disabled by default for backward compatibility
}

fn default_window_size() -> usize {
    5
}

fn default_interpolation_method() -> InterpolationMethod {
    InterpolationMethod::Linear
}

impl Default for BoundarySmoothingConfig {
    fn default() -> Self {
        Self {
            enabled: default_smoothing_enabled(),
            window_size: default_window_size(),
            interpolation: default_interpolation_method(),
        }
    }
}

impl BoundarySmoothingConfig {
    /// Validate configuration values
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `window_size` is not in range [1, 20]
    ///
    /// # Examples
    ///
    /// ```
    /// use arrow_quant_v2::BoundarySmoothingConfig;
    ///
    /// let config = BoundarySmoothingConfig::default();
    /// config.validate().expect("Invalid configuration");
    /// ```
    pub fn validate(&self) -> Result<()> {
        if !(1..=20).contains(&self.window_size) {
            return Err(QuantError::ConfigurationError(format!(
                "window_size must be in range [1, 20], got {}",
                self.window_size
            )));
        }
        Ok(())
    }
}

/// Beta schedule type for diffusion process (Phase 3)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BetaSchedule {
    /// Linear schedule: beta_t = beta_start + (beta_end - beta_start) * t / T
    Linear,
    /// Cosine schedule: beta_t = 1 - alpha_t, where alpha_t follows cosine curve
    Cosine,
}

impl Default for BetaSchedule {
    fn default() -> Self {
        BetaSchedule::Linear
    }
}

impl From<BetaSchedule> for crate::thermodynamic::BetaSchedule {
    fn from(schedule: BetaSchedule) -> Self {
        match schedule {
            BetaSchedule::Linear => crate::thermodynamic::BetaSchedule::Linear,
            BetaSchedule::Cosine => crate::thermodynamic::BetaSchedule::Cosine,
        }
    }
}

/// Configuration for transition optimization (Phase 3)
///
/// Optimizes quantization parameters to preserve transition probabilities
/// and minimize thermodynamic loss. This is the most expensive phase but
/// provides the largest accuracy improvement (+4-5% cumulative).
///
/// # Examples
///
/// ```
/// use arrow_quant_v2::{TransitionOptimizationConfig, BetaSchedule};
///
/// let config = TransitionOptimizationConfig {
///     enabled: true,
///     markov_weight: 0.1,
///     entropy_weight: 0.05,
///     learning_rate: 0.01,
///     max_iterations: 50,
///     convergence_threshold: 1e-4,
///     beta_schedule: BetaSchedule::Linear,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionOptimizationConfig {
    /// Enable transition optimization
    #[serde(default = "default_optimization_enabled")]
    pub enabled: bool,

    /// Weight for Markov constraint loss (default: 0.1)
    /// Higher values enforce stronger Markov property preservation
    #[serde(default = "default_markov_weight")]
    pub markov_weight: f32,

    /// Weight for entropy regularization (default: 0.05)
    /// Encourages diverse quantization values
    #[serde(default = "default_entropy_weight")]
    pub entropy_weight: f32,

    /// Learning rate for gradient descent (default: 0.01)
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f32,

    /// Maximum optimization iterations (default: 50)
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,

    /// Convergence threshold for early stopping (default: 1e-4)
    #[serde(default = "default_convergence_threshold")]
    pub convergence_threshold: f32,

    /// Beta schedule type for diffusion process
    #[serde(default)]
    pub beta_schedule: BetaSchedule,
}

fn default_optimization_enabled() -> bool {
    false // Disabled by default (expensive operation)
}

fn default_markov_weight() -> f32 {
    0.1
}

fn default_entropy_weight() -> f32 {
    0.05
}

fn default_learning_rate() -> f32 {
    0.01
}

fn default_max_iterations() -> usize {
    50
}

fn default_convergence_threshold() -> f32 {
    1e-4
}

impl Default for TransitionOptimizationConfig {
    fn default() -> Self {
        Self {
            enabled: default_optimization_enabled(),
            markov_weight: default_markov_weight(),
            entropy_weight: default_entropy_weight(),
            learning_rate: default_learning_rate(),
            max_iterations: default_max_iterations(),
            convergence_threshold: default_convergence_threshold(),
            beta_schedule: BetaSchedule::default(),
        }
    }
}

impl TransitionOptimizationConfig {
    /// Validate configuration values
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `markov_weight` is negative
    /// - `entropy_weight` is negative
    /// - `learning_rate` is not in range (0.0, 1.0]
    /// - `max_iterations` is 0
    /// - `convergence_threshold` is not positive
    pub fn validate(&self) -> Result<()> {
        if self.markov_weight < 0.0 {
            return Err(QuantError::ConfigurationError(format!(
                "markov_weight must be non-negative, got {}",
                self.markov_weight
            )));
        }
        if self.entropy_weight < 0.0 {
            return Err(QuantError::ConfigurationError(format!(
                "entropy_weight must be non-negative, got {}",
                self.entropy_weight
            )));
        }
        if self.learning_rate <= 0.0 || self.learning_rate > 1.0 {
            return Err(QuantError::ConfigurationError(format!(
                "learning_rate must be in range (0.0, 1.0], got {}",
                self.learning_rate
            )));
        }
        if self.max_iterations == 0 {
            return Err(QuantError::ConfigurationError(
                "max_iterations must be greater than 0".to_string()
            ));
        }
        if self.convergence_threshold <= 0.0 {
            return Err(QuantError::ConfigurationError(format!(
                "convergence_threshold must be positive, got {}",
                self.convergence_threshold
            )));
        }
        Ok(())
    }
}

/// Configuration for thermodynamic constraints
///
/// Provides configuration for all three phases of thermodynamic enhancement:
/// - Phase 1: Markov validation (monitoring only)
/// - Phase 2: Boundary smoothing (+2-3% accuracy)
/// - Phase 3: Transition optimization (+4-5% cumulative accuracy)
///
/// All features are opt-in and backward compatible.
///
/// # Examples
///
/// ```
/// use arrow_quant_v2::ThermodynamicConfig;
///
/// // Default: validation enabled in debug mode, smoothing and optimization disabled
/// let config = ThermodynamicConfig::default();
///
/// // Custom configuration with all phases enabled
/// let config = ThermodynamicConfig {
///     validation: arrow_quant_v2::ValidationConfig {
///         enabled: true,
///         smoothness_threshold: 0.25,
///         log_violations: true,
///     },
///     boundary_smoothing: arrow_quant_v2::BoundarySmoothingConfig {
///         enabled: true,
///         window_size: 5,
///         interpolation: arrow_quant_v2::InterpolationMethod::Linear,
///     },
///     transition_optimization: arrow_quant_v2::TransitionOptimizationConfig {
///         enabled: true,
///         markov_weight: 0.1,
///         entropy_weight: 0.05,
///         learning_rate: 0.01,
///         max_iterations: 50,
///         convergence_threshold: 1e-4,
///         beta_schedule: arrow_quant_v2::BetaSchedule::Linear,
///     },
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicConfig {
    /// Phase 1: Markov validation configuration
    #[serde(default)]
    pub validation: ValidationConfig,

    /// Phase 2: Boundary smoothing configuration
    #[serde(default)]
    pub boundary_smoothing: BoundarySmoothingConfig,

    /// Phase 3: Transition optimization configuration
    #[serde(default)]
    pub transition_optimization: TransitionOptimizationConfig,
}

impl Default for ThermodynamicConfig {
    fn default() -> Self {
        Self {
            validation: ValidationConfig::default(),
            boundary_smoothing: BoundarySmoothingConfig::default(),
            transition_optimization: TransitionOptimizationConfig::default(),
        }
    }
}

impl ThermodynamicConfig {
    /// Validate all thermodynamic configuration values
    ///
    /// # Errors
    ///
    /// Returns an error if any sub-configuration is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use arrow_quant_v2::ThermodynamicConfig;
    ///
    /// let config = ThermodynamicConfig::default();
    /// config.validate().expect("Invalid configuration");
    /// ```
    pub fn validate(&self) -> Result<()> {
        self.validation.validate()?;
        self.boundary_smoothing.validate()?;
        self.transition_optimization.validate()?;
        Ok(())
    }
}

impl Default for DiffusionQuantConfig {
    fn default() -> Self {
        Self::from_profile(DeploymentProfile::Local)
    }
}

impl DiffusionQuantConfig {
    /// Load configuration from YAML file
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use arrow_quant_v2::DiffusionQuantConfig;
    ///
    /// let config = DiffusionQuantConfig::from_yaml("config.yaml")
    ///     .expect("Failed to load config");
    /// ```
    pub fn from_yaml<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            QuantError::ConfigurationError(format!("Failed to read config file: {}", e))
        })?;

        let config: Self = serde_yaml::from_str(&content)
            .map_err(|e| QuantError::ConfigurationError(format!("Failed to parse YAML: {}", e)))?;

        // Validate after loading
        config.validate()?;

        Ok(config)
    }

    /// Apply environment variable overrides
    ///
    /// Supported environment variables:
    /// - `ARROW_QUANT_BIT_WIDTH`: Override bit_width (2, 4, or 8)
    /// - `ARROW_QUANT_NUM_TIME_GROUPS`: Override num_time_groups
    /// - `ARROW_QUANT_GROUP_SIZE`: Override group_size (32, 64, 128, or 256)
    /// - `ARROW_QUANT_MIN_ACCURACY`: Override min_accuracy (0.0-1.0)
    /// - `ARROW_QUANT_CALIBRATION_SAMPLES`: Override calibration_samples
    /// - `ARROW_QUANT_FAIL_FAST`: Override fail_fast (true/false)
    /// - `ARROW_QUANT_NUM_THREADS`: Override num_threads (0 = auto-detect)
    /// - `ARROW_QUANT_ENABLE_STREAMING`: Override enable_streaming (true/false)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use arrow_quant_v2::DiffusionQuantConfig;
    ///
    /// let mut config = DiffusionQuantConfig::default();
    /// config.apply_env_overrides();
    /// ```
    pub fn apply_env_overrides(&mut self) {
        if let Ok(val) = std::env::var("ARROW_QUANT_BIT_WIDTH") {
            if let Ok(bit_width) = val.parse::<u8>() {
                self.bit_width = bit_width;
            }
        }

        if let Ok(val) = std::env::var("ARROW_QUANT_NUM_TIME_GROUPS") {
            if let Ok(num_groups) = val.parse::<usize>() {
                self.num_time_groups = num_groups;
            }
        }

        if let Ok(val) = std::env::var("ARROW_QUANT_GROUP_SIZE") {
            if let Ok(group_size) = val.parse::<usize>() {
                self.group_size = group_size;
            }
        }

        if let Ok(val) = std::env::var("ARROW_QUANT_MIN_ACCURACY") {
            if let Ok(min_accuracy) = val.parse::<f32>() {
                self.min_accuracy = min_accuracy;
            }
        }

        if let Ok(val) = std::env::var("ARROW_QUANT_CALIBRATION_SAMPLES") {
            if let Ok(samples) = val.parse::<usize>() {
                self.calibration_samples = samples;
            }
        }

        if let Ok(val) = std::env::var("ARROW_QUANT_FAIL_FAST") {
            if let Ok(fail_fast) = val.parse::<bool>() {
                self.fail_fast = fail_fast;
            }
        }

        if let Ok(val) = std::env::var("ARROW_QUANT_NUM_THREADS") {
            if let Ok(num_threads) = val.parse::<usize>() {
                self.num_threads = num_threads;
            }
        }

        if let Ok(val) = std::env::var("ARROW_QUANT_ENABLE_STREAMING") {
            if let Ok(enable_streaming) = val.parse::<bool>() {
                self.enable_streaming = enable_streaming;
            }
        }

        if let Ok(val) = std::env::var("ARROW_QUANT_ENABLE_MEMORY_AWARE") {
            if let Ok(enable) = val.parse::<bool>() {
                self.enable_memory_aware_scheduling = enable;
            }
        }

        if let Ok(val) = std::env::var("ARROW_QUANT_MAX_MEMORY_MB") {
            if let Ok(limit) = val.parse::<usize>() {
                self.max_memory_limit_mb = Some(limit);
            }
        }
    }

    /// Save configuration to YAML file
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use arrow_quant_v2::{DiffusionQuantConfig, DeploymentProfile};
    ///
    /// let config = DiffusionQuantConfig::from_profile(DeploymentProfile::Edge);
    /// config.to_yaml("config.yaml").expect("Failed to save config");
    /// ```
    pub fn to_yaml<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let yaml = serde_yaml::to_string(self).map_err(|e| {
            QuantError::ConfigurationError(format!("Failed to serialize config: {}", e))
        })?;

        std::fs::write(path.as_ref(), yaml).map_err(|e| {
            QuantError::ConfigurationError(format!("Failed to write config file: {}", e))
        })?;

        Ok(())
    }

    /// Create configuration from deployment profile
    pub fn from_profile(profile: DeploymentProfile) -> Self {
        match profile {
            DeploymentProfile::Edge => Self {
                bit_width: 2,
                modality: None,
                num_time_groups: 5,
                group_size: 256,
                enable_time_aware: true,
                enable_spatial: false,
                min_accuracy: 0.65,
                calibration_samples: 32,
                deployment_profile: profile,
                fail_fast: false,
                num_threads: 0,         // Auto-detect
                enable_streaming: true, // Enable for memory-constrained devices
                skip_sensitive_layers: false,
                sensitive_layer_names: Vec::new(),
                sensitive_layer_patterns: Vec::new(),
                enable_mixed_precision: false,
                enable_entropy_adaptation: false,
                layer_bit_widths: std::collections::HashMap::new(),
                target_model_size_mb: None,
                enable_memory_aware_scheduling: true,
                max_memory_limit_mb: None,
                thermodynamic: ThermodynamicConfig::default(),
            },
            DeploymentProfile::Local => Self {
                bit_width: 4,
                modality: None,
                num_time_groups: 10,
                group_size: 128,
                enable_time_aware: true,
                enable_spatial: true,
                min_accuracy: 0.85,
                calibration_samples: 128,
                deployment_profile: profile,
                fail_fast: false,
                num_threads: 0,          // Auto-detect
                enable_streaming: true,  // Streaming mode to prevent OOM
                skip_sensitive_layers: false,
                sensitive_layer_names: Vec::new(),
                sensitive_layer_patterns: Vec::new(),
                enable_mixed_precision: false,
                enable_entropy_adaptation: false,
                layer_bit_widths: std::collections::HashMap::new(),
                target_model_size_mb: None,
                enable_memory_aware_scheduling: true,
                max_memory_limit_mb: None,
                thermodynamic: ThermodynamicConfig::default(),
            },
            DeploymentProfile::Cloud => Self {
                bit_width: 8,
                modality: None,
                num_time_groups: 20,
                group_size: 64,
                enable_time_aware: true,
                enable_spatial: true,
                min_accuracy: 0.95,
                calibration_samples: 512,
                deployment_profile: profile,
                fail_fast: false,
                num_threads: 0,          // Auto-detect
                enable_streaming: false, // Batch mode for better performance
                skip_sensitive_layers: false,
                sensitive_layer_names: Vec::new(),
                sensitive_layer_patterns: Vec::new(),
                enable_mixed_precision: false,
                enable_entropy_adaptation: false,
                layer_bit_widths: std::collections::HashMap::new(),
                target_model_size_mb: None,
                enable_memory_aware_scheduling: true,
                max_memory_limit_mb: None,
                thermodynamic: ThermodynamicConfig::default(),
            },
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if ![2, 4, 8].contains(&self.bit_width) {
            return Err(QuantError::InvalidBitWidth(self.bit_width));
        }

        if self.num_time_groups < 1 || self.num_time_groups > 100 {
            return Err(QuantError::InvalidTimeGroups(self.num_time_groups));
        }

        if ![32, 64, 128, 256].contains(&self.group_size) {
            return Err(QuantError::InvalidGroupSize(self.group_size));
        }

        if !(0.0..=1.0).contains(&self.min_accuracy) {
            return Err(QuantError::InvalidAccuracy(self.min_accuracy));
        }

        // Validate thermodynamic configuration
        self.thermodynamic.validate()?;

        Ok(())
    }

    /// Create base mode configuration (no diffusion enhancements)
    pub fn base_mode() -> Self {
        Self {
            bit_width: 4,
            modality: None,
            num_time_groups: 1,
            group_size: 128,
            enable_time_aware: false,
            enable_spatial: false,
            min_accuracy: 0.90,
            calibration_samples: 128,
            deployment_profile: DeploymentProfile::Local,
            fail_fast: false,
            num_threads: 0, // Auto-detect
            enable_streaming: false,
            skip_sensitive_layers: false,
            sensitive_layer_names: Vec::new(),
            sensitive_layer_patterns: Vec::new(),
            enable_mixed_precision: false,
            enable_entropy_adaptation: false,
            layer_bit_widths: std::collections::HashMap::new(),
            target_model_size_mb: None,
            enable_memory_aware_scheduling: false,
            max_memory_limit_mb: None,
            thermodynamic: ThermodynamicConfig::default(),
        }
    }

    /// Set per-layer bit-width for mixed-precision quantization
    ///
    /// # Arguments
    ///
    /// * `layer_name` - Name of the layer (e.g., "model.layers.0.self_attn.q_proj")
    /// * `bit_width` - Target bit-width (2, 4, 8, or 16 for FP16)
    ///
    /// # Examples
    ///
    /// ```
    /// use arrow_quant_v2::DiffusionQuantConfig;
    ///
    /// let mut config = DiffusionQuantConfig::default();
    /// config.enable_mixed_precision = true;
    ///
    /// // Use INT2 for most layers (default)
    /// config.bit_width = 2;
    ///
    /// // Use INT4 for attention layers (more sensitive)
    /// config.set_layer_bit_width("model.layers.0.self_attn.q_proj", 4);
    /// config.set_layer_bit_width("model.layers.0.self_attn.k_proj", 4);
    ///
    /// // Preserve FP16 for embeddings
    /// config.set_layer_bit_width("model.embed_tokens", 16);
    /// ```
    pub fn set_layer_bit_width(&mut self, layer_name: &str, bit_width: u8) {
        self.layer_bit_widths.insert(layer_name.to_string(), bit_width);
    }

    /// Get bit-width for a specific layer
    ///
    /// Returns the layer-specific bit-width if set, otherwise returns the default bit_width.
    ///
    /// # Arguments
    ///
    /// * `layer_name` - Name of the layer
    ///
    /// # Returns
    ///
    /// The bit-width to use for this layer (2, 4, 8, or 16)
    pub fn get_layer_bit_width(&self, layer_name: &str) -> u8 {
        if !self.enable_mixed_precision {
            return self.bit_width;
        }

        self.layer_bit_widths
            .get(layer_name)
            .copied()
            .unwrap_or(self.bit_width)
    }

    /// Analyze layer sensitivity and assign optimal bit-widths automatically
    ///
    /// This method analyzes the model structure and assigns bit-widths based on:
    /// 1. Layer type (embeddings, attention, MLP, norms, output heads)
    /// 2. Layer depth (early vs late layers)
    /// 3. Target model size constraint
    ///
    /// # Strategy
    ///
    /// - **Embeddings & Output Heads**: FP16 (most sensitive)
    /// - **Layer Norms**: FP16 (critical for stability)
    /// - **Attention Layers**: INT4 (moderately sensitive)
    /// - **Early Layers (0-25%)**: INT4 (more important for quality)
    /// - **Middle Layers (25-75%)**: INT2 (can tolerate more quantization)
    /// - **Late Layers (75-100%)**: INT4 (important for final output)
    ///
    /// # Arguments
    ///
    /// * `layer_names` - List of all layer names in the model
    /// * `layer_sizes` - Size of each layer in bytes (for size optimization)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use arrow_quant_v2::DiffusionQuantConfig;
    /// use std::collections::HashMap;
    ///
    /// let mut config = DiffusionQuantConfig::default();
    /// config.enable_mixed_precision = true;
    /// config.target_model_size_mb = Some(100.0); // Target 100MB
    ///
    /// let layer_names = vec![
    ///     "model.embed_tokens.weight",
    ///     "model.layers.0.self_attn.q_proj.weight",
    ///     "model.layers.0.mlp.gate_proj.weight",
    ///     "lm_head.weight",
    /// ];
    ///
    /// let layer_sizes: HashMap<String, usize> = layer_names
    ///     .iter()
    ///     .map(|name| (name.to_string(), 1024 * 1024)) // 1MB each
    ///     .collect();
    ///
    /// config.analyze_and_assign_bit_widths(&layer_names, &layer_sizes);
    /// ```
    pub fn analyze_and_assign_bit_widths(
        &mut self,
        layer_names: &[String],
        layer_sizes: &std::collections::HashMap<String, usize>,
    ) {
        if !self.enable_mixed_precision {
            return;
        }

        let total_layers = layer_names.len();

        for (idx, layer_name) in layer_names.iter().enumerate() {
            let layer_lower = layer_name.to_lowercase();

            // Strategy 1: Sensitive layers get FP16
            if self.is_sensitive_layer_name(&layer_lower) {
                self.set_layer_bit_width(layer_name, 16);
                continue;
            }

            // Strategy 2: Attention layers get INT4 (moderately sensitive)
            if layer_lower.contains("attn") || layer_lower.contains("attention") {
                self.set_layer_bit_width(layer_name, 4);
                continue;
            }

            // Strategy 3: Layer depth-based assignment
            let layer_position = idx as f32 / total_layers as f32;

            if layer_position < 0.25 {
                // Early layers (0-25%): INT4 (more important)
                self.set_layer_bit_width(layer_name, 4);
            } else if layer_position < 0.75 {
                // Middle layers (25-75%): INT2 (can tolerate more quantization)
                self.set_layer_bit_width(layer_name, 2);
            } else {
                // Late layers (75-100%): INT4 (important for output)
                self.set_layer_bit_width(layer_name, 4);
            }
        }

        // Strategy 4: Optimize for target model size if specified
        if let Some(target_size_mb) = self.target_model_size_mb {
            self.optimize_for_target_size(layer_names, layer_sizes, target_size_mb);
        }
    }

    /// Helper method to check if a layer name matches sensitive patterns
    fn is_sensitive_layer_name(&self, layer_lower: &str) -> bool {
        let sensitive_patterns = [
            "embed", "embedding", ".wte.", ".wpe.", "norm", "ln_", "layernorm", "lm_head",
            ".head.", "output", "pooler",
        ];

        sensitive_patterns
            .iter()
            .any(|pattern| layer_lower.contains(pattern))
    }

    /// Optimize bit-width assignments to meet target model size
    ///
    /// This method adjusts bit-widths to meet the target model size constraint while
    /// preserving quality as much as possible. It uses a greedy approach:
    /// 1. Calculate current model size with assigned bit-widths
    /// 2. If over target, reduce bit-widths for least sensitive layers
    /// 3. If under target, increase bit-widths for most sensitive layers
    fn optimize_for_target_size(
        &mut self,
        layer_names: &[String],
        layer_sizes: &std::collections::HashMap<String, usize>,
        target_size_mb: f32,
    ) {
        let target_size_bytes = (target_size_mb * 1024.0 * 1024.0) as usize;

        // Calculate current size
        let mut current_size: usize = 0;
        for layer_name in layer_names {
            let layer_size = layer_sizes.get(layer_name).copied().unwrap_or(0);
            let bit_width = self.get_layer_bit_width(layer_name);
            let quantized_size = (layer_size * bit_width as usize) / 16; // Assuming FP16 baseline
            current_size += quantized_size;
        }

        // If we're over target, reduce bit-widths for non-sensitive layers
        if current_size > target_size_bytes {
            // Sort layers by size (largest first) and reduce bit-widths
            let mut layer_info: Vec<_> = layer_names
                .iter()
                .map(|name| {
                    let size = layer_sizes.get(name).copied().unwrap_or(0);
                    let bit_width = self.get_layer_bit_width(name);
                    (name.clone(), size, bit_width)
                })
                .collect();

            layer_info.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by size descending

            for (layer_name, layer_size, current_bit_width) in layer_info {
                if current_size <= target_size_bytes {
                    break;
                }

                // Skip sensitive layers
                if current_bit_width == 16 {
                    continue;
                }

                // Try to reduce bit-width
                let new_bit_width = match current_bit_width {
                    8 => 4,
                    4 => 2,
                    _ => continue, // Already at minimum
                };

                let size_saved =
                    (layer_size * (current_bit_width - new_bit_width) as usize) / 16;
                self.set_layer_bit_width(&layer_name, new_bit_width);
                current_size -= size_saved;
            }
        }
    }
}

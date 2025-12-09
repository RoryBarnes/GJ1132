import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
import vplot

sPlotFile = "EngleAgeHist.png"
iNumBins = 50
dFigSizeX = 3.25
dFigSizeY = 3
xmin=-0.1
xmax=1.063
ymin=0
ymax=0.06
dMaxLogAge = np.log10(13)  # 13 Gyr

def ComputeAgeDistribution(daA, daB, daC, daD, daRotationPeriod, iNumSamples=100000):

    daASamples = np.random.normal(daA[0], daA[1], iNumSamples)
    daBSamples = np.random.normal(daB[0], daB[1], iNumSamples)
    daCSamples = np.random.normal(daC[0], daC[1], iNumSamples)
    daDSamples = np.random.normal(daD[0], daD[1], iNumSamples)
    daRotPerSamples = np.random.normal(daRotationPeriod[0], daRotationPeriod[1], iNumSamples)
    
    daLogAge = daASamples * daRotPerSamples + daBSamples + daCSamples * (daRotPerSamples - daDSamples)
    daLogAge = daLogAge[daLogAge <= dMaxLogAge]
    
    print(f"Samples after filtering (≤ 12 Gyr): {len(daLogAge):,} out of {iNumSamples:,} original samples")
    print(f"Percentage retained: {100 * len(daLogAge) / iNumSamples:.1f}%")

    dMeanAge = np.mean(daLogAge)
    dStdDevAge = np.std(daLogAge)
    daAge95ConfidenceInterval = np.percentile(daLogAge, [2.5, 97.5])
    
    return daLogAge, dMeanAge, dStdDevAge, daAge95ConfidenceInterval

def AnalyticalSolution(daA, daB, daC, daD, daRotationPeriod):
    dMeanA, dStdDevA = daA
    dMeanB, dStdDevB = daB
    dMeanC, dStdDevC = daC
    dMeanD, dStdDevD = daD
    dRotPerMean, dStdDevRotPer = daRotationPeriod
    
    dMeanAgeAnalytic = (dMeanA + dMeanC) * dRotPerMean + dMeanB - dMeanC * dMeanD
    
    return dMeanAgeAnalytic

def NormalizedHistogram(daColumn,color,label=None):
    fig = plt.figure(figsize=(dFigSizeX, dFigSizeY))
    counts, bin_edges = np.histogram(daColumn, bins=iNumBins)
    daNormalizedFractions = counts / len(daColumn)
    plt.step(bin_edges[:-1], daNormalizedFractions,where='mid', color=color,label=label)
    plt.xlabel('Log(Age) [Gyr]',fontsize=12)
    plt.ylabel('Fraction',fontsize=12)
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    #plt.legend(loc='best')
    plt.tight_layout()

    #plt.show()
    plt.savefig(sPlotFile,dpi=300)

def plot_results(y_samples, y_mean, y_std, y_ci):
    """Plot histogram of y values with statistics."""
    plt.figure(figsize=(6.5, 6))
    
    # Histogram
    plt.subplot(2, 1, 1)
    plt.hist(y_samples, bins=100, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(y_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {y_mean:.3f}')
    plt.axvline(y_ci[0], color='orange', linestyle=':', linewidth=2, label=f'95% CI: [{y_ci[0]:.3f}, {y_ci[1]:.3f}]')
    plt.axvline(y_ci[1], color='orange', linestyle=':', linewidth=2)
    plt.axvline(np.log10(13), color='green', linestyle='-', linewidth=2, label='Constraint: Age ≤ 13 Gyr')
    plt.xlabel('Log(Age) [Gyr]')
    plt.ylabel('Probability Density')
    plt.title('Distribution of Stellar Ages [Filtered: Age ≤ 13 Gyr]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Q-Q plot to check normality
    plt.subplot(2, 1, 2)
    stats.probplot(y_samples, dist="norm", plot=plt)
    plt.title('Q-Q Plot: Checking if Log(Age) follows Normal Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    daA = (0.0251, 0.0018)
    daB = (-0.1615, 0.0303)
    daC = (-0.0212, 0.0018)
    daD = (25.45, 1.9079)
    daRotationPeriod = (122, 5.5)
    

    daLogAge, dMeanAge, dStdDevAge, daAge95ConfidenceInterval = ComputeAgeDistribution(
        daA, daB, daC, daD, daRotationPeriod
    )
    
    dMeanAgeAnalytic = AnalyticalSolution(daA, daB, daC, daD, daRotationPeriod)
    
    print(f"\nMonte Carlo Results ({len(daLogAge):,} samples after filtering):")
    print(f"Best fit (mean): {dMeanAge:.4f}")
    print(f"Uncertainty (std): {dStdDevAge:.4f}")
    print(f"95% Confidence Interval: [{daAge95ConfidenceInterval[0]:.4f}, {daAge95ConfidenceInterval[1]:.4f}]")
    print(f"Range of plausible values: [{np.min(daLogAge):.4f}, {np.max(daLogAge):.4f}] (all ≤ {np.log10(13):.3f})")
    print(f"\nAnalytical mean (for comparison): {dMeanAgeAnalytic:.4f}")
    print(f"Difference between Monte Carlo and analytical: {abs(dMeanAge - dMeanAgeAnalytic):.6f}")
    
    # Additional statistics
    print(f"\nAdditional Statistics:")
    print(f"Median: {np.median(daLogAge):.4f}")
    try:
        mode_result = stats.mode(np.round(daLogAge, 2))
        if hasattr(mode_result, 'mode'):
            # Newer scipy versions
            mode_value = mode_result.mode
        else:
            # Older scipy versions
            mode_value = mode_result[0]
        
        # Handle both scalar and array results
        if np.isscalar(mode_value):
            print(f"Mode (approximate): {mode_value:.2f}")
        else:
            print(f"Mode (approximate): {mode_value[0]:.2f}")
    except (IndexError, AttributeError):
        # Fallback: use histogram-based mode estimation
        hist, bin_edges = np.histogram(daLogAge, bins=50)
        mode_bin = np.argmax(hist)
        mode_estimate = (bin_edges[mode_bin] + bin_edges[mode_bin + 1]) / 2
        print(f"Mode (approximate): {mode_estimate:.2f}")

    #print(f"Mode (approximate): {stats.mode(np.round(daLogAge, 2))[0][0]:.2f}")
    print(f"Skewness: {stats.skew(daLogAge):.4f}")
    print(f"Kurtosis: {stats.kurtosis(daLogAge):.4f}")
    
    # Additional check for maximum constraint
    dMaxAge = np.max(daLogAge)
    print(f"\nConstraint Check:")
    print(f"Maximum age value (after filtering): {dMaxAge:.4f}")
    print(f"Maximum age constraint (log10(13)): {np.log10(13):.4f}")
    
    print(f"\nFull Range of Valid Age Values:")
    print(f"Minimum age: {np.min(daLogAge):.4f}")
    print(f"Maximum age: {dMaxAge:.4f}")
    dShapiroStatistic, dShapiroP = stats.shapiro(daLogAge[:5000])  # Use subset for efficiency
    print(f"Shapiro-Wilk test p-value: {dShapiroP:.6f}")
    print(f"Is Age normally distributed? {'Yes' if dShapiroP > 0.05 else 'No'}")
    
    NormalizedHistogram(daLogAge,'k',label='')
    #plot_results(daLogAge, dMeanAge, dStdDevAge, daAge95ConfidenceInterval)
    
    # Convert to linear age in years (only for ages that passed the filter)
    daAge = 10**daLogAge
    daAge *= 1e9  # Convert from Gyr to years

    sOutFile='age_samples.txt'
    np.savetxt(sOutFile, daAge)
    print(f"\nResults saved to '{sOutFile}'")
    print(f"Age range in output file: {np.min(daAge)/1e9:.2f} - {np.max(daAge)/1e9:.2f} Gyr")

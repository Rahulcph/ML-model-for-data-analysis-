# ML-model-for-data-analysis-
A machine learning model for analyzing participation and engagement data in cybersecurity awareness events. Includes data preprocessing, feature selection, model training (decision tree), and result visualization.

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)

print("Libraries imported successfully!")
# Read CSV file
df = pd.read_csv('ML_Event.csv')

# Show the first 5 rows
print(df.head())
# Clean and preprocess the data
print("🔧 Data Preprocessing")
print("-" * 40)

# Convert participation columns to binary (1 for Yes, 0 for No)
participation_cols = ['Participation in Event 1', 'Participation in Event 2', 'Participation in Event 3']
for col in participation_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Create simplified column names for easier analysis
df['Event_1'] = df['Participation in Event 1']
df['Event_2'] = df['Participation in Event 2']
df['Event_3'] = df['Participation in Event 3']

# Calculate total participation for each person
df['Total_Participation'] = df[['Event_1', 'Event_2', 'Event_3']].sum(axis=1)
df['Participation_Rate'] = (df['Total_Participation'] / 3) * 100

print(f"Average participation rate: {df['Participation_Rate'].mean():.1f}%")
print(f"Total participants: {len(df)}")
print(f"Unique job profiles: {df['Job Profile'].nunique()}")
print(f"Unique locations: {df['Location'].nunique()}")
print(f"Unique organizations: {df['Organization'].nunique()}")

# Display processed data sample
display_cols = ['Job Profile', 'Location', 'Event_1', 'Event_2', 'Event_3', 'Total_Participation', 'Organization', 'Experience Level']
df[display_cols].head(10)
# CELL 4: Overall Event Participation Analysis
# ============================================================================

print("📊 MAXIMUM PARTICIPATION ANALYSIS BY EVENT")
print("=" * 50)

# Calculate participation by event
event_participation = {
    'Event 1': df['Event_1'].sum(),
    'Event 2': df['Event_2'].sum(),
    'Event 3': df['Event_3'].sum()
}

print("\n🎯 Overall Event Participation:")
for event, count in event_participation.items():
    percentage = (count / len(df)) * 100
    print(f"{event}: {count} participants ({percentage:.1f}%)")

# Find maximum participation event
max_event = max(event_participation, key=event_participation.get)
max_count = event_participation[max_event]
print(f"\n🏆 Highest Participation: {max_event} with {max_count} participants")

# Visualize overall event participation
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
events = list(event_participation.keys())
counts = list(event_participation.values())
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

bars = ax.bar(events, counts, color=colors)
ax.set_title('Overall Participation by Event', fontsize=16, fontweight='bold')
ax.set_ylabel('Number of Participants')
ax.set_xlabel('Events')

# Add value labels on bars
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{count}\n({(count/len(df)*100):.1f}%)',
            ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()
CELL 5: Maximum Participation by Job Profile
# ============================================================================

print("💼 MAXIMUM PARTICIPATION BY JOB PROFILE")
print("=" * 45)

# Calculate participation by job profile for each event
job_event_participation = df.groupby('Job Profile')[['Event_1', 'Event_2', 'Event_3']].sum()

print("\n📋 Participation by Job Profile and Event:")
print(job_event_participation.to_string())

# Find maximum participation per event by job profile
print("\n🏆 TOP JOB PROFILES PER EVENT:")
for i, event in enumerate(['Event_1', 'Event_2', 'Event_3'], 1):
    max_job = job_event_participation[event].idxmax()
    max_count = job_event_participation[event].max()
    print(f"Event {i}: {max_job} ({max_count} participants)")
# Calculate total participation by job profile
job_total = job_event_participation.sum(axis=1).sort_values(ascending=False)
print(f"\n🌟 Overall Top Job Profile: {job_total.index[0]} ({job_total.iloc[0]} total participants)")
CELL 6: Visualize Job Profile Participation
# ============================================================================

# Visualize job profile participation
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Participation Analysis by Job Profile', fontsize=16, fontweight='bold')

# Event 1 participation by job profile
event1_data = job_event_participation['Event_1'].sort_values(ascending=False)
axes[0,0].bar(range(len(event1_data)), event1_data.values, color='#FF6B6B')
axes[0,0].set_title('Event 1 Participation by Job Profile')
axes[0,0].set_xticks(range(len(event1_data)))
axes[0,0].set_xticklabels(event1_data.index, rotation=45, ha='right')
axes[0,0].set_ylabel('Participants')

# Event 2 participation by job profile
event2_data = job_event_participation['Event_2'].sort_values(ascending=False)
axes[0,1].bar(range(len(event2_data)), event2_data.values, color='#4ECDC4')
axes[0,1].set_title('Event 2 Participation by Job Profile')
axes[0,1].set_xticks(range(len(event2_data)))
axes[0,1].set_xticklabels(event2_data.index, rotation=45, ha='right')
axes[0,1].set_ylabel('Participants')
Event 3 participation by job profile
event3_data = job_event_participation['Event_3'].sort_values(ascending=False)
axes[1,0].bar(range(len(event3_data)), event3_data.values, color='#45B7D1')
axes[1,0].set_title('Event 3 Participation by Job Profile')
axes[1,0].set_xticks(range(len(event3_data)))
axes[1,0].set_xticklabels(event3_data.index, rotation=45, ha='right')
axes[1,0].set_ylabel('Participants')

# Heatmap of job profile participation across events
im = axes[1,1].imshow(job_event_participation.T.values, cmap='YlOrRd', aspect='auto')
axes[1,1].set_title('Job Profile Participation Heatmap')
axes[1,1].set_xticks(range(len(job_event_participation.index)))
axes[1,1].set_xticklabels(job_event_participation.index, rotation=45, ha='right')
axes[1,1].set_yticks(range(3))
axes[1,1].set_yticklabels(['Event 1', 'Event 2', 'Event 3'])

# Add colorbar
cbar = plt.colorbar(im, ax=axes[1,1])
cbar.set_label('Number of Participants')

plt.tight_layout()
plt.show()
 CELL 7: Maximum Participation by Location
# ============================================================================

print("🌍 MAXIMUM PARTICIPATION BY LOCATION")
print("=" * 40)

# Calculate participation by location for each event
location_event_participation = df.groupby('Location')[['Event_1', 'Event_2', 'Event_3']].sum()

print("\n📍 Participation by Location and Event:")
print(location_event_participation.to_string())

# Find maximum participation per event by location
print("\n🏆 TOP LOCATIONS PER EVENT:")
for i, event in enumerate(['Event_1', 'Event_2', 'Event_3'], 1):
    max_location = location_event_participation[event].idxmax()
    max_count = location_event_participation[event].max()
    print(f"Event {i}: {max_location} ({max_count} participants)")

# Calculate total participation by location
location_total = location_event_participation.sum(axis=1).sort_values(ascending=False)
print(f"\n🌟 Overall Top Location: {location_total.index[0]} ({location_total.iloc[0]} total participants)")
 CELL 8: Visualize Location Participation
# ============================================================================

# Visualize location participation
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart of total participation by location
axes[0].bar(range(len(location_total)), location_total.values, 
           color=plt.cm.viridis(np.linspace(0, 1, len(location_total))))
axes[0].set_title('Total Participation by Location', fontweight='bold')
axes[0].set_xticks(range(len(location_total)))
axes[0].set_xticklabels(location_total.index, rotation=45, ha='right')
axes[0].set_ylabel('Total Participants')

# Add value labels
for i, v in enumerate(location_total.values):
    axes[0].text(i, v + 0.05, str(v), ha='center', va='bottom', fontweight='bold')

# Line plot showing participation trend by location across events
for location in location_event_participation.index:
 values = location_event_participation.loc[location].values
    axes[1].plot([1, 2, 3], values, marker='o', linewidth=2, label=location)

axes[1].set_title('Participation Trend by Location Across Events', fontweight='bold')
axes[1].set_xlabel('Event Number')
axes[1].set_ylabel('Number of Participants')
axes[1].set_xticks([1, 2, 3])
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
ELL 9: Analysis by Organization and Experience Level
# ============================================================================

print("🏢 PARTICIPATION BY ORGANIZATION AND EXPERIENCE LEVEL")
print("=" * 55)

# Organization analysis
org_participation = df.groupby('Organization')[['Event_1', 'Event_2', 'Event_3']].sum()
org_total = org_participation.sum(axis=1).sort_values(ascending=False)

print("\n🏢 Top Organizations by Total Participation:")
for org, count in org_total.head().items():
    print(f"{org}: {count} total participants")

# Experience level analysis
exp_participation = df.groupby('Experience Level')[['Event_1', 'Event_2', 'Event_3']].sum()
exp_total = exp_participation.sum(axis=1).sort_values(ascending=False)

print("\n🎓 Participation by Experience Level:")
for level, count in exp_total.items():
    print(f"{level}: {count} total participants")

# Find maximum per event
print("\n🏆 TOP ORGANIZATIONS PER EVENT:")
for i, event in enumerate(['Event_1', 'Event_2', 'Event_3'], 1):
    max_org = org_participation[event].idxmax()
    max_count = org_participation[event].max()
    print(f"Event {i}: {max_org} ({max_count} participants)")

print("\n🏆 TOP EXPERIENCE LEVELS PER EVENT:")
for i, event in enumerate(['Event_1', 'Event_2', 'Event_3'], 1):
    max_exp = exp_participation[event].idxmax()
    max_count = exp_participation[event].max()
    print(f"Event {i}: {max_exp} ({max_count} participants)")
CELL 10: Visualize Organization and Experience Analysis
# ============================================================================

# Visualize organization and experience analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Organization and Experience Level Analysis', fontsize=16, fontweight='bold')

# Organization participation
axes[0,0].bar(range(len(org_total)), org_total.values, color='#2ECC71')
axes[0,0].set_title('Total Participation by Organization')
axes[0,0].set_xticks(range(len(org_total)))
axes[0,0].set_xticklabels(org_total.index, rotation=45, ha='right')
axes[0,0].set_ylabel('Total Participants')

# Experience level participation
axes[0,1].bar(range(len(exp_total)), exp_total.values, color='#E74C3C')
axes[0,1].set_title('Total Participation by Experience Level')
axes[0,1].set_xticks(range(len(exp_total)))
axes[0,1].set_xticklabels(exp_total.index)
axes[0,1].set_ylabel('Total Participants')

# Organization trend across events
for org in org_participation.index:
    values = org_participation.loc[org].values
axes[1,0].plot([1, 2, 3], values, marker='o', linewidth=2, label=org)
axes[1,0].set_title('Organization Participation Trend')
axes[1,0].set_xlabel('Event Number')
axes[1,0].set_ylabel('Participants')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Experience level trend across events
for level in exp_participation.index:
    values = exp_participation.loc[level].values
    axes[1,1].plot([1, 2, 3], values, marker='o', linewidth=2, label=level)
axes[1,1].set_title('Experience Level Participation Trend')
axes[1,1].set_xlabel('Event Number')
axes[1,1].set_ylabel('Participants')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
CELL 11: Summary Statistics and Key Insights
# ============================================================================

print("📈 SUMMARY STATISTICS AND KEY INSIGHTS")
print("=" * 45)

# Overall statistics
total_participants = len(df)
event_stats = {
    'Event 1': df['Event_1'].sum(),
    'Event 2': df['Event_2'].sum(), 
    'Event 3': df['Event_3'].sum()
}

print("\n📊 OVERALL STATISTICS:")
print(f"Total Participants: {total_participants}")
print(f"Average Participation Rate: {df['Participation_Rate'].mean():.1f}%")

for event, count in event_stats.items():
    percentage = (count / total_participants) * 100
    print(f"{event}: {count} participants ({percentage:.1f}%)")

# Find maximum participation patterns
print("\n🏆 MAXIMUM PARTICIPATION PATTERNS:")

job_max = job_event_participation.max(axis=1).sort_values(ascending=False)
print(f"\nTop Job Profile (single event): {job_max.index[0]} ({job_max.iloc[0]} participants)")

job_total = job_event_participation.sum(axis=1).sort_values(ascending=False)
print(f"Top Job Profile (total): {job_total.index[0]} ({job_total.iloc[0]} total participants)")

# By Location
location_max = location_event_participation.max(axis=1).sort_values(ascending=False)
print(f"\nTop Location (single event): {location_max.index[0]} ({location_max.iloc[0]} participants)")
print(f"Top Location (total): {location_total.index[0]} ({location_total.iloc[0]} total participants)")

# By Organization
print(f"\nTop Organization: {org_total.index[0]} ({org_total.iloc[0]} total participants)")

# By Experience Level
print(f"Top Experience Level: {exp_total.index[0]} ({exp_total.iloc[0]} total participants)")

# Participation trends
print("\n📈 PARTICIPATION TRENDS:")
if event_stats['Event 3'] > event_stats['Event 1']:
    print("✅ Participation is INCREASING over time")
elif event_stats['Event 3'] < event_stats['Event 1']:
print("⚠️ Participation is DECREASING over time")
else:
    print("➡️ Participation is STABLE over time")

print(f"\nEvent with highest participation: {max(event_stats, key=event_stats.get)}")
print(f"Event with lowest participation: {min(event_stats, key=event_stats.get)}")
CELL 12: Custom Analysis Functions
# ============================================================================

# Custom functions for detailed analysis

def analyze_participation_by_category(category_column, top_n=5):
    """
    Analyze participation by any categorical column
    """
    print(f"\n📊 PARTICIPATION ANALYSIS BY {category_column.upper()}")
    print("=" * 50)
    
    # Calculate participation by category
    category_participation = df.groupby(category_column)[['Event_1', 'Event_2', 'Event_3']].sum()
    category_total = category_participation.sum(axis=1).sort_values(ascending=False)
    
    print(f"\nTop {top_n} {category_column}s by total participation:")
    for i, (category, count) in enumerate(category_total.head(top_n).items()):
        print(f"{i+1}. {category}: {count} participants")
     return category_participation, category_total

def find_max_participation_per_event(category_column):
    """
    Find maximum participation per event for a given category
    """
    category_participation = df.groupby(category_column)[['Event_1', 'Event_2', 'Event_3']].sum()
    
    print(f"\n🏆 MAXIMUM PARTICIPATION PER EVENT BY {category_column.upper()}:")
    for i, event in enumerate(['Event_1', 'Event_2', 'Event_3'], 1):
        max_category = category_participation[event].idxmax()
        max_count = category_participation[event].max()
        print(f"Event {i}: {max_category} ({max_count} participants)")
    
    return category_participation

print("🚀 QUICK ANALYSIS COMMANDS")
print("=" * 30)

# Run these for quick insights:
print("\n1. Maximum participation by Job Profile:")
print(f"   {job_event_participation.sum(axis=1).idxmax()}: {job_event_participation.sum(axis=1).max()} total")

print("\n2. Maximum participation by Location:")
print(f"   {location_event_participation.sum(axis=1).idxmax()}: {location_event_participation.sum(axis=1).max()} total")

print("\n3. Maximum participation by Organization:")
print(f"   {org_total.index[0]}: {org_total.iloc[0]} total")

print("\n4. Maximum participation by Experience Level:")
print(f"   {exp_total.index[0]}: {exp_total.iloc[0]} total")

print("\n5. Event with highest overall participation:")
highest_event = max(event_stats, key=event_stats.get)
print(f"   {highest_event}: {event_stats[highest_event]} participants")

# Create summary dataframe for easy export
summary_df = pd.DataFrame({
    'Category': ['Job Profile', 'Location', 'Organization', 'Experience Level'],
    'Top_Performer': [
        job_event_participation.sum(axis=1).idxmax(),
        location_event_participation.sum(axis=1).idxmax(), 
        org_total.index[0],
        exp_total.index[0]
    ],
    'Max_Participation': [
        job_event_participation.sum(axis=1).max(),
        location_event_participation.sum(axis=1).max(),
        org_total.iloc[0],
        exp_total.iloc[0]
    ]
})

print("\n📋 SUMMARY TABLE:")
print(summary_df.to_string(index=False))

# Save summary if needed
# summary_df.to_csv('participation_summary.csv', index=False)

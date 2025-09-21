#!/usr/bin/env python3
"""
Enhanced Dataset Generator V3 - High Diversity & Banking Standards Compliant
Author: Vũ Tuấn Chiến
Description: Generate 10,000 unique, diverse test cases for mobile banking
"""

import json
import random
import hashlib
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Set
from datetime import datetime
from collections import defaultdict, Counter
import re

class DiverseBankingTestGenerator:
    """Generate highly diverse test cases for mobile banking with security focus"""
    
    def __init__(self, schema_path: str = None):
        # Load schema if provided
        if schema_path and Path(schema_path).exists():
            with open(schema_path, 'r') as f:
                self.schema = yaml.safe_load(f)
        else:
            self.schema = self._get_default_schema()
        
        # Initialize diverse data pools
        self._initialize_data_pools()
        
        # Track uniqueness
        self.generated_hashes = set()
        self.generated_patterns = set()
        self.step_variations = defaultdict(list)
        
    def _get_default_schema(self) -> Dict:
        """Default schema based on ai_optimized_schema_v2_with_security.yaml"""
        return {
            'features': {
                'supported': [
                    'login_authentication', 'fund_transfer', 'bill_payment',
                    'account_balance', 'transaction_history', 'card_management',
                    'profile_management', 'beneficiary_management',
                    'notification_settings', 'support_help'
                ],
                'distribution': {
                    'login_authentication': 0.12,
                    'fund_transfer': 0.15,
                    'bill_payment': 0.10,
                    'account_balance': 0.08,
                    'transaction_history': 0.08,
                    'card_management': 0.10,
                    'profile_management': 0.08,
                    'beneficiary_management': 0.12,
                    'notification_settings': 0.07,
                    'support_help': 0.10
                }
            },
            'scenario_types': {
                'positive': {'weight': 0.35},
                'negative': {'weight': 0.20},
                'security': {'weight': 0.25},
                'edge': {'weight': 0.20}
            }
        }
    
    def _initialize_data_pools(self):
        """Initialize diverse data pools for variation"""
        
        # Vietnamese names
        self.vn_first_names = [
            'Nguyễn Văn', 'Trần Thị', 'Lê Văn', 'Phạm Thị', 'Hoàng Văn',
            'Huỳnh Thị', 'Phan Văn', 'Võ Thị', 'Đặng Văn', 'Bùi Thị',
            'Đỗ Văn', 'Hồ Thị', 'Ngô Văn', 'Dương Thị', 'Lý Văn'
        ]
        
        self.vn_last_names = [
            'An', 'Bình', 'Cường', 'Dũng', 'Đức', 'Giang', 'Hải', 
            'Hùng', 'Khang', 'Long', 'Minh', 'Nam', 'Phúc', 'Quân',
            'Sơn', 'Tài', 'Thành', 'Thiện', 'Trung', 'Vinh'
        ]
        
        # Account numbers with varied formats
        self.account_formats = [
            lambda: f"{random.randint(1000,9999)}{random.randint(1000,9999)}{random.randint(1000,9999)}",
            lambda: f"{random.randint(100,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}",
            lambda: f"VCB{random.randint(10000000,99999999)}",
            lambda: f"ACB-{random.randint(1000,9999)}-{random.randint(100000,999999)}",
            lambda: f"MB{random.randint(100,999)}{random.randint(100000,999999)}"
        ]
        
        # Amounts with realistic variations
        self.amount_pools = {
            'small': [100_000, 200_000, 300_000, 500_000, 750_000],
            'medium': [1_000_000, 2_000_000, 5_000_000, 8_000_000],
            'large': [10_000_000, 20_000_000, 50_000_000, 100_000_000],
            'exact': [999_999, 1_000_001, 4_999_999, 10_000_001],
            'decimal': [150_500, 2_350_750, 5_555_555, 12_345_678]
        }
        
        # Bills and services
        self.bill_types = [
            'Electricity', 'Water', 'Internet', 'Mobile', 'Cable TV',
            'Gas', 'Insurance', 'School Fee', 'Hospital', 'Parking'
        ]
        
        self.bill_providers = {
            'Electricity': ['EVN HCMC', 'EVN Hanoi', 'EVN Central', 'PC1', 'PC2', 'PC3'],
            'Water': ['Sawaco', 'Hawaco', 'Dawaco', 'BWC', 'TDW'],
            'Internet': ['FPT', 'Viettel', 'VNPT', 'CMC', 'NetNam', 'SPT'],
            'Mobile': ['Viettel', 'Mobifone', 'Vinaphone', 'Vietnamobile', 'Gmobile']
        }
        
        # Security test variations
        self.security_patterns = {
            'authentication': [
                'session timeout after inactivity',
                'multiple failed login attempts',
                'password complexity validation',
                '2FA code verification',
                'biometric authentication',
                'device registration check'
            ],
            'data_privacy': [
                'PII data masking',
                'encrypted transmission',
                'audit log generation',
                'consent management',
                'data retention policy',
                'GDPR compliance check'
            ],
            'payment_security': [
                'CVV verification',
                'transaction signing',
                'SSL/TLS validation',
                'tokenization check',
                'fraud detection',
                'velocity checking'
            ]
        }
        
        # UI elements variations
        self.ui_elements = {
            'buttons': ['Submit', 'Confirm', 'Cancel', 'Next', 'Back', 'Continue', 'Proceed'],
            'fields': ['amount field', 'account field', 'password field', 'OTP field', 'description field'],
            'pages': ['login page', 'dashboard', 'transfer screen', 'payment page', 'profile section'],
            'messages': ['success message', 'error alert', 'confirmation dialog', 'warning popup']
        }
        
        # Time variations
        self.time_contexts = [
            'during business hours',
            'after midnight',
            'on weekend',
            'on public holiday',
            'at month end',
            'during maintenance window'
        ]
        
        # Device contexts
        self.device_contexts = [
            'on iPhone 14',
            'on Samsung Galaxy S23',
            'on iPad',
            'on Android tablet',
            'with slow network',
            'in airplane mode',
            'with VPN enabled'
        ]
    
    def generate_unique_test_id(self) -> str:
        """Generate unique test ID"""
        timestamp = datetime.now().strftime("%Y%m%d")
        random_num = random.randint(1000, 9999)
        return f"TC_{timestamp}_{random_num}"
    
    def generate_diverse_preconditions(self, feature: str, scenario_type: str) -> List[str]:
        """Generate diverse preconditions based on context"""
        preconditions = []
        
        # Base precondition with variation
        app_states = [
            "User has mobile banking app installed",
            "Mobile banking app is updated to latest version",
            "User has registered for mobile banking",
            "Mobile banking service is activated"
        ]
        preconditions.append(random.choice(app_states))
        
        # Feature-specific preconditions
        if feature == 'login_authentication':
            preconditions.append(random.choice([
                "User has valid credentials",
                "User account is active",
                "User has registered device",
                "2FA is enabled for account"
            ]))
        elif feature == 'fund_transfer':
            amount = random.choice(random.choice(list(self.amount_pools.values())))
            preconditions.append(f"Account balance is {amount:,} VND")
            preconditions.append(random.choice([
                "Daily transfer limit is set",
                "Beneficiary is already added",
                "Transaction password is set up"
            ]))
        elif feature == 'bill_payment':
            bill_type = random.choice(self.bill_types)
            preconditions.append(f"{bill_type} bill is due")
            preconditions.append("Payment method is configured")
        
        # Security-specific preconditions
        if scenario_type == 'security':
            preconditions.append(random.choice([
                "Security monitoring is active",
                "SSL certificate is valid",
                "Audit logging is enabled",
                "Encryption is configured"
            ]))
        
        # Context variations
        if random.random() > 0.7:
            preconditions.append(f"Testing {random.choice(self.device_contexts)}")
        
        if random.random() > 0.8:
            preconditions.append(f"Test executed {random.choice(self.time_contexts)}")
        
        return preconditions[:random.randint(2, 4)]  # Return 2-4 preconditions
    
    def generate_diverse_steps(self, feature: str, scenario_type: str) -> List[str]:
        """Generate diverse test steps with high variation"""
        steps = []
        
        # Generate unique Given step
        given_variations = self._generate_given_step(feature)
        steps.append(given_variations)
        
        # Add context with And steps (0-2 random)
        for _ in range(random.randint(0, 2)):
            and_step = self._generate_and_step(feature, scenario_type)
            if and_step and and_step not in steps:
                steps.append(and_step)
        
        # Generate unique When step
        when_step = self._generate_when_step(feature, scenario_type)
        steps.append(when_step)
        
        # Add more And steps for actions (1-3 random)
        for _ in range(random.randint(1, 3)):
            and_step = self._generate_action_and_step(feature, scenario_type)
            if and_step and and_step not in steps:
                steps.append(and_step)
        
        # Generate unique Then step
        then_step = self._generate_then_step(feature, scenario_type)
        steps.append(then_step)
        
        # Add verification And steps (0-2 random)
        for _ in range(random.randint(0, 2)):
            and_step = self._generate_verification_and_step(feature, scenario_type)
            if and_step and and_step not in steps:
                steps.append(and_step)
        
        return steps
    
    def _generate_given_step(self, feature: str) -> str:
        """Generate diverse Given steps"""
        templates = {
            'login_authentication': [
                f"Given user is on {random.choice(['login page', 'authentication screen', 'sign-in interface'])}",
                f"Given user opens mobile banking app {random.choice(self.device_contexts)}",
                f"Given user navigates to login section"
            ],
            'fund_transfer': [
                f"Given user is on {random.choice(['transfer page', 'fund transfer screen', 'payment section'])}",
                f"Given user has selected transfer feature",
                f"Given user is viewing transfer options"
            ],
            'bill_payment': [
                f"Given user is on bill payment {random.choice(['page', 'screen', 'section'])}",
                f"Given user navigates to {random.choice(self.bill_types)} payment",
                f"Given user opens payment center"
            ]
        }
        
        default_templates = [
            f"Given user is logged into mobile banking",
            f"Given user has accessed {feature.replace('_', ' ')} feature",
            f"Given user is on {random.choice(['main dashboard', 'home screen', 'menu page'])}"
        ]
        
        return random.choice(templates.get(feature, default_templates))
    
    def _generate_and_step(self, feature: str, scenario_type: str) -> str:
        """Generate context And steps"""
        contexts = []
        
        # Add security context for security scenarios
        if scenario_type == 'security':
            contexts.extend([
                "And security monitoring is active",
                "And SSL connection is verified",
                "And session is being tracked",
                "And audit logging is enabled"
            ])
        
        # Add data context
        if feature in ['fund_transfer', 'bill_payment']:
            amount = random.choice(random.choice(list(self.amount_pools.values())))
            contexts.extend([
                f"And account balance shows {amount:,} VND",
                f"And daily limit is set to {amount*10:,} VND",
                "And transaction history is loaded"
            ])
        
        # Add UI state context
        contexts.extend([
            f"And {random.choice(self.ui_elements['buttons'])} button is visible",
            "And all required fields are displayed",
            "And page loads successfully"
        ])
        
        return random.choice(contexts) if contexts else ""
    
    def _generate_when_step(self, feature: str, scenario_type: str) -> str:
        """Generate diverse When steps"""
        
        if feature == 'login_authentication':
            username = f"{random.choice(self.vn_first_names)}_{random.choice(self.vn_last_names)}"
            actions = [
                f"When user enters username '{username}'",
                f"When user provides credentials",
                f"When user attempts to login",
                f"When user submits authentication form"
            ]
            
        elif feature == 'fund_transfer':
            account = random.choice(self.account_formats)()
            amount = random.choice(random.choice(list(self.amount_pools.values())))
            actions = [
                f"When user enters amount {amount:,} VND",
                f"When user selects beneficiary account {account}",
                f"When user initiates transfer",
                f"When user confirms transaction details"
            ]
            
        elif feature == 'bill_payment':
            bill = random.choice(self.bill_types)
            provider = random.choice(self.bill_providers.get(bill, ['Provider']))
            actions = [
                f"When user selects {bill} bill from {provider}",
                f"When user enters bill amount",
                f"When user chooses payment method",
                f"When user confirms payment"
            ]
        else:
            actions = [
                f"When user performs {feature.replace('_', ' ')} action",
                f"When user interacts with {feature.replace('_', ' ')}",
                f"When user executes requested operation"
            ]
        
        # Add negative scenarios
        if scenario_type == 'negative':
            actions.extend([
                f"When user enters invalid data",
                f"When user exceeds maximum limit",
                f"When user provides incorrect information"
            ])
        
        return random.choice(actions)
    
    def _generate_action_and_step(self, feature: str, scenario_type: str) -> str:
        """Generate action-oriented And steps"""
        actions = []
        
        # Common actions
        actions.extend([
            f"And clicks {random.choice(self.ui_elements['buttons'])} button",
            f"And enters required information",
            f"And reviews the details",
            f"And scrolls to bottom of page"
        ])
        
        # Security actions
        if scenario_type == 'security' or feature == 'login_authentication':
            otp = f"{random.randint(100000, 999999)}"
            actions.extend([
                f"And enters OTP code '{otp}'",
                "And completes 2FA verification",
                "And provides biometric authentication",
                "And enters transaction password"
            ])
        
        # Feature specific
        if feature == 'fund_transfer':
            actions.extend([
                "And selects transfer type",
                "And adds transfer description",
                "And chooses fee payment option",
                "And confirms beneficiary details"
            ])
        
        return random.choice(actions) if actions else ""
    
    def _generate_then_step(self, feature: str, scenario_type: str) -> str:
        """Generate diverse Then steps"""
        
        if scenario_type == 'positive':
            results = [
                f"Then {feature.replace('_', ' ')} is successful",
                f"Then user sees {random.choice(self.ui_elements['messages'])}",
                f"Then operation completes successfully",
                f"Then system processes request correctly"
            ]
            
        elif scenario_type == 'negative':
            error_codes = ['ERR_001', 'ERR_AUTH_01', 'ERR_LIMIT_02', 'ERR_VAL_03']
            results = [
                f"Then error message '{random.choice(error_codes)}' is displayed",
                f"Then transaction is rejected",
                f"Then validation error appears",
                f"Then system blocks the operation"
            ]
            
        elif scenario_type == 'security':
            results = [
                "Then security validation passes",
                "Then audit log entry is created",
                "Then session is secured",
                "Then encryption is verified"
            ]
        else:  # edge
            results = [
                "Then system handles edge case correctly",
                "Then boundary condition is processed",
                "Then special case is managed properly"
            ]
        
        return random.choice(results)
    
    def _generate_verification_and_step(self, feature: str, scenario_type: str) -> str:
        """Generate verification And steps"""
        verifications = []
        
        # Common verifications
        verifications.extend([
            "And confirmation number is generated",
            "And transaction record is saved",
            "And notification is sent",
            "And balance is updated"
        ])
        
        # Security verifications
        if scenario_type == 'security':
            verifications.extend([
                "And security log is updated",
                "And session timeout is set",
                "And encryption status is confirmed",
                "And compliance check passes"
            ])
        
        # Feature specific
        if feature == 'fund_transfer':
            verifications.extend([
                "And transfer receipt is available",
                "And beneficiary receives notification",
                "And transaction appears in history"
            ])
        
        return random.choice(verifications) if verifications else ""
    
    def generate_test_case(self, feature: str = None, scenario_type: str = None) -> Dict:
        """Generate a single unique test case"""
        
        # Select feature and scenario if not provided
        if not feature:
            features = self.schema['features']['supported']
            weights = [self.schema['features']['distribution'].get(f, 0.1) for f in features]
            feature = random.choices(features, weights=weights)[0]
        
        if not scenario_type:
            scenarios = list(self.schema['scenario_types'].keys())
            weights = [self.schema['scenario_types'][s]['weight'] for s in scenarios]
            scenario_type = random.choices(scenarios, weights=weights)[0]
        
        # Generate test case components
        test_id = self.generate_unique_test_id()
        title = self._generate_unique_title(feature, scenario_type)
        preconditions = self.generate_diverse_preconditions(feature, scenario_type)
        test_steps = self.generate_diverse_steps(feature, scenario_type)
        expected_results = self._generate_expected_results(feature, scenario_type)
        
        # Build test case
        test_case = {
            'test_id': test_id,
            'title': title,
            'feature': feature,
            'scenario_type': scenario_type,
            'priority': self._determine_priority(feature, scenario_type),
            'preconditions': preconditions,
            'test_steps': test_steps,
            'expected_results': expected_results,
            'quality_score': self._calculate_quality_score(test_steps, scenario_type)
        }
        
        # Add security fields if applicable
        if scenario_type == 'security':
            test_case['security_validations'] = self._generate_security_validations(feature)
            test_case['compliance_checks'] = self._generate_compliance_checks(feature)
            test_case['risk_level'] = random.choice(['HIGH', 'MEDIUM', 'LOW'])
        
        # Check uniqueness
        test_hash = hashlib.md5(json.dumps(test_steps).encode()).hexdigest()
        if test_hash in self.generated_hashes:
            # Regenerate if duplicate
            return self.generate_test_case(feature, scenario_type)
        
        self.generated_hashes.add(test_hash)
        return test_case
    
    def _generate_unique_title(self, feature: str, scenario_type: str) -> str:
        """Generate unique descriptive title"""
        
        action_verbs = {
            'positive': ['Verify', 'Test', 'Validate', 'Check', 'Confirm'],
            'negative': ['Test invalid', 'Verify error for', 'Check rejection of', 'Validate failure'],
            'security': ['Security test for', 'Validate security of', 'Check compliance for'],
            'edge': ['Edge case for', 'Boundary test for', 'Special case of']
        }
        
        feature_descriptions = {
            'login_authentication': ['user login', 'authentication process', 'sign-in flow'],
            'fund_transfer': ['money transfer', 'fund movement', 'payment transfer'],
            'bill_payment': ['bill payment', 'utility payment', 'invoice settlement']
        }
        
        verb = random.choice(action_verbs.get(scenario_type, ['Test']))
        desc = random.choice(feature_descriptions.get(feature, [feature.replace('_', ' ')]))
        
        # Add unique context
        contexts = [
            f"with {random.choice(['valid', 'invalid', 'special'])} data",
            f"during {random.choice(self.time_contexts)}",
            f"{random.choice(self.device_contexts)}",
            f"with {random.choice(['normal', 'high', 'low'])} amount"
        ]
        
        return f"{verb} {desc} {random.choice(contexts)}"
    
    def _determine_priority(self, feature: str, scenario_type: str) -> str:
        """Determine test priority based on feature and scenario"""
        
        critical_features = ['login_authentication', 'fund_transfer', 'payment_security']
        high_features = ['bill_payment', 'card_management', 'beneficiary_management']
        
        if scenario_type == 'security' or feature in critical_features:
            return 'Critical'
        elif feature in high_features or scenario_type == 'negative':
            return 'High'
        elif scenario_type == 'positive':
            return 'Medium'
        else:
            return 'Low'
    
    def _generate_expected_results(self, feature: str, scenario_type: str) -> List[str]:
        """Generate expected results"""
        results = []
        
        if scenario_type == 'positive':
            results.append(f"{feature.replace('_', ' ').title()} completes successfully")
            results.append("Success message is displayed")
            results.append("User is redirected to appropriate page")
            
        elif scenario_type == 'negative':
            results.append("Appropriate error message is shown")
            results.append("Transaction/Operation is blocked")
            results.append("System state remains unchanged")
            
        elif scenario_type == 'security':
            results.append("Security validation passes")
            results.append("Audit log entry is created")
            results.append("Compliance requirements are met")
            
        else:  # edge
            results.append("Edge case is handled gracefully")
            results.append("System remains stable")
            results.append("No data corruption occurs")
        
        # Add specific validations
        if feature == 'fund_transfer':
            results.append("Balance is updated correctly")
            results.append("Transaction ID is generated")
            
        return results[:random.randint(2, 4)]
    
    def _generate_security_validations(self, feature: str) -> List[str]:
        """Generate security validation points"""
        validations = []
        
        # Common validations
        validations.extend([
            "SSL/TLS encryption is active",
            "Session token is valid",
            "Input data is sanitized"
        ])
        
        # Feature specific
        if feature == 'login_authentication':
            validations.extend([
                "Password is hashed",
                "Login attempts are tracked",
                "Session timeout is configured"
            ])
        elif feature in ['fund_transfer', 'bill_payment']:
            validations.extend([
                "Transaction is signed",
                "Amount limits are enforced",
                "Fraud detection is triggered"
            ])
        
        return random.sample(validations, min(3, len(validations)))
    
    def _generate_compliance_checks(self, feature: str) -> List[str]:
        """Generate compliance check points"""
        checks = []
        
        standards = ['ISO-27001', 'GDPR', 'PCI-DSS', 'MASVS']
        
        for standard in random.sample(standards, random.randint(1, 3)):
            if standard == 'GDPR':
                checks.append(f"{standard}: User consent is obtained")
            elif standard == 'PCI-DSS':
                checks.append(f"{standard}: Card data is tokenized")
            elif standard == 'ISO-27001':
                checks.append(f"{standard}: Access control is enforced")
            else:
                checks.append(f"{standard}: Mobile security verified")
        
        return checks
    
    def _calculate_quality_score(self, test_steps: List[str], scenario_type: str) -> float:
        """Calculate quality score for test case"""
        score = 0.4  # Base score
        
        # Check for clear steps
        if all(any(keyword in step for keyword in ['Given', 'When', 'Then', 'And']) for step in test_steps):
            score += 0.15
        
        # Check for completeness
        if len(test_steps) >= 4:
            score += 0.15
        
        # Security bonus
        if scenario_type == 'security':
            score += 0.20
        
        # Diversity bonus
        if len(set(test_steps)) == len(test_steps):  # All steps unique
            score += 0.10
        
        return min(1.0, score)
    
    def generate_dataset(self, total_samples: int = 10000) -> List[Dict]:
        """Generate complete dataset with high diversity"""
        dataset = []
        
        # Distribution from schema
        distribution = {
            'positive': int(total_samples * 0.35),
            'negative': int(total_samples * 0.20),
            'security': int(total_samples * 0.25),
            'edge': int(total_samples * 0.20)
        }
        
        print(f"🚀 Generating {total_samples} diverse test cases...")
        print(f"   Distribution: {distribution}")
        
        for scenario_type, count in distribution.items():
            print(f"\n📝 Generating {count} {scenario_type} scenarios...")
            
            for i in range(count):
                if (i + 1) % 500 == 0:
                    print(f"   Progress: {i+1}/{count}")
                
                # Retry logic for uniqueness
                max_retries = 5
                for retry in range(max_retries):
                    try:
                        test_case = self.generate_test_case(scenario_type=scenario_type)
                        dataset.append(test_case)
                        break
                    except Exception as e:
                        if retry == max_retries - 1:
                            print(f"   Warning: Could not generate unique case after {max_retries} retries")
        
        print(f"\n✅ Generated {len(dataset)} test cases")
        print(f"   Unique cases: {len(self.generated_hashes)}")
        
        return dataset
    
    def calculate_dataset_metrics(self, dataset: List[Dict]) -> Dict:
        """Calculate diversity metrics for dataset"""
        
        # Extract all steps
        all_steps = []
        for item in dataset:
            all_steps.extend(item['test_steps'])
        
        # Calculate metrics
        unique_steps = len(set(all_steps))
        total_steps = len(all_steps)
        
        # Feature distribution
        feature_counts = Counter(item['feature'] for item in dataset)
        
        # Scenario distribution
        scenario_counts = Counter(item['scenario_type'] for item in dataset)
        
        # Quality scores
        quality_scores = [item['quality_score'] for item in dataset]
        
        return {
            'total_cases': len(dataset),
            'unique_hashes': len(self.generated_hashes),
            'total_steps': total_steps,
            'unique_steps': unique_steps,
            'diversity_score': unique_steps / total_steps if total_steps > 0 else 0,
            'feature_distribution': dict(feature_counts),
            'scenario_distribution': dict(scenario_counts),
            'avg_quality_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            'security_coverage': scenario_counts.get('security', 0) / len(dataset) if dataset else 0
        }

def main():
    """Main function to generate dataset"""
    
    # Initialize generator
    generator = DiverseBankingTestGenerator()
    
    # Generate dataset
    dataset = generator.generate_dataset(total_samples=10000)
    
    # Calculate metrics
    metrics = generator.calculate_dataset_metrics(dataset)
    
    # Print metrics
    print("\n" + "="*60)
    print("📊 DATASET METRICS")
    print("="*60)
    print(f"Total cases: {metrics['total_cases']:,}")
    print(f"Unique cases: {metrics['unique_hashes']:,}")
    print(f"Diversity score: {metrics['diversity_score']:.3f}")
    print(f"Avg quality: {metrics['avg_quality_score']:.2f}")
    print(f"Security coverage: {metrics['security_coverage']:.1%}")
    
    # Save dataset
    output_dir = Path(__file__).parent.parent.parent / 'datasets/current'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Split dataset
    train_size = int(len(dataset) * 0.7)
    val_size = int(len(dataset) * 0.15)
    
    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size+val_size]
    test_data = dataset[train_size+val_size:]
    
    # Save files
    files = {
        f'train_diverse_v3_{timestamp}.json': train_data,
        f'val_diverse_v3_{timestamp}.json': val_data,
        f'test_diverse_v3_{timestamp}.json': test_data
    }
    
    for filename, data in files.items():
        filepath = output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved {filename}: {len(data)} cases")
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'metrics': metrics,
        'schema_version': 'v2_with_security',
        'generator': 'DiverseBankingTestGenerator_V3'
    }
    
    with open(output_dir / f'metadata_diverse_v3_{timestamp}.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✨ Dataset generation complete!")
    print(f"📁 Output directory: {output_dir}")

if __name__ == "__main__":
    main()

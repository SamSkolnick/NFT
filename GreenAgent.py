import pandas as pd
from ResearchEval import evaluate_research

class GreenAgent:
    def __init__(self, task_config):
        self.task_data_path = task_config['data_path']
        self.constraints = task_config['constraints']  # time, memory, etc.
        self.test_labels = task_config['test_labels']  # Hidden from white agent
        
    def evaluate(self, submission):
        """
        submission = {
            'docker_image': 'white_agent_v1:latest',
            'research_artifacts': '/path/to/research/',  # Papers, notes, experiments
            'code': '/path/to/code/'  # Optional, for inspection,
            'storage_method': method for research
        }
        """
        results = {}
        
        # 1. Evaluate Research (BEFORE running solution)
        results['research'] = evaluate_research(
            submission['research_artifacts'],
            storage=submission['storage_method']
        )
        
        # 2. Run Solution
        execution = self.run_white_agent(
            submission['docker_image'],
            self.task_data_path
        )
        
        # 3. Check Constraints
        results['constraints'] = self.check_constraints(
            execution,
            self.constraints
        )
        
        # 4. Evaluate Performance
        if results['constraints']['passed']:
            results['performance'] = self.evaluate_performance(
                execution['predictions'],
                self.test_labels
            )
        else:
            results['performance'] = 0.0  # Failed constraints
            
        return results
    
    def evaluate_research(self, research_path, task_path):
        """
        THIS is where green agent needs to be agentic
        """
        # Use Claude Agent SDK to read and evaluate research
        options = ClaudeAgentOptions(
            cwd=research_path,
            allowed_tools=["Read", "Grep", "WebFetch"],  # Can verify citations
            system_prompt=f"""Evaluate the research quality for this ML task.
            
            Task description is at {task_path}/task_description.json
            
            Check:
            1. Did they research relevant methods?
            2. Are citations real and applicable?
            3. Did they run experiments to inform their approach?
            4. Is there evidence of understanding domain constraints?
            
            Return JSON with scores and justification.
            """
        )
        
        # Run agentic evaluation
        async with ClaudeSDKClient(options=options) as client:
            await client.query("Evaluate the research artifacts in this directory")
            evaluation = await self.parse_evaluation(client)
            
        return evaluation
    
    def run_white_agent(self, docker_image, task_path):
        """
        Non-agentic: just run container and collect results
        """
        import docker
        client = docker.from_env()
        
        # Create output directory
        output_path = f"/tmp/outputs_{uuid.uuid4()}"
        os.makedirs(output_path)
        
        # Run with resource limits
        container = client.containers.run(
            docker_image,
            volumes={
                f"{task_path}/train": {'bind': '/data/train', 'mode': 'ro'},
                f"{task_path}/val": {'bind': '/data/val', 'mode': 'ro'},
                output_path: {'bind': '/output', 'mode': 'rw'}
            },
            mem_limit="8g",
            cpu_quota=200000,  # 2 CPUs
            detach=True,
            remove=False  # Keep for inspection if needed
        )
        
        # Wait with timeout
        start_time = time.time()
        try:
            result = container.wait(timeout=3600)  # 1 hour
            elapsed_time = time.time() - start_time
            
            # Get resource usage
            stats = container.stats(stream=False)
            
            return {
                'success': result['StatusCode'] == 0,
                'predictions': f"{output_path}/predictions.csv",
                'time_seconds': elapsed_time,
                'memory_used_mb': stats['memory_stats']['usage'] / 1024 / 1024,
                'logs': container.logs().decode('utf-8')
            }
        except docker.errors.ContainerError as e:
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            container.remove()
    
    def check_constraints(self, execution, constraints):
        """
        Non-agentic: simple checks
        """
        if not execution['success']:
            return {'passed': False, 'reason': 'Execution failed'}
            
        violations = []
        
        if execution['time_seconds'] > constraints['max_time_seconds']:
            violations.append(f"Time limit exceeded: {execution['time_seconds']}s > {constraints['max_time_seconds']}s")
            
        if execution['memory_used_mb'] > constraints['max_memory_mb']:
            violations.append(f"Memory limit exceeded: {execution['memory_used_mb']}MB > {constraints['max_memory_mb']}MB")
            
        # Check output format
        try:
            preds = pd.read_csv(execution['predictions'])
            if not self.valid_format(preds):
                violations.append("Invalid prediction format")
        except:
            violations.append("Could not read predictions")
            
        return {
            'passed': len(violations) == 0,
            'violations': violations
        }
    
    def evaluate_performance(self, predictions_path, test_labels):
        """
        Non-agentic: compute metrics
        """
        preds = pd.read_csv(predictions_path)
        
        from sklearn.metrics import accuracy_score, f1_score
        
        return {
            'accuracy': accuracy_score(test_labels, preds['prediction']),
            'f1_score': f1_score(test_labels, preds['prediction'], average='weighted')
        } 
from libs.models import RepoCrawlTarget

# If you add a subnet here it will get crawled
#   'url': github url e.g. 'https://github.com/corcel-api/cortex.t',
#   'branch': e.g. 'main',
#   'name': human readable name e.g. 'cortex.t',
#   'target_collection': name of vector db collection, e.g. 'subnet18'

_crawl_targets = {
    'subnet-18': {
        'url': 'https://github.com/corcel-api/cortex.t',
        'branch': 'main',
        'name': 'cortex.t',
        'target_collection': 'subnet18',
        'tag': 'subnet-18: platform for AI development and synthetic data generation'
    },
    'subnet-19': {
        'url': 'https://github.com/namoray/vision',
        'branch': 'main',
        'name': 'vision',
        'target_collection': 'subnet19',
        'tag': 'subnet-19: decentralised inference of AI models'
    },
    'myself': {
        'url': 'https://github.com/radu-mutilica/chat-with-repo',
        'branch': 'master',
        'name': 'chat-with-repo',
        'target_collection': 'myself',
        'tag': 'AI chatbot for talking to GitHub repositories'
    }
    # 'subnet-1': 'https://github.com/macrocosm-os/prompting',
    # 'subnet-4': 'https://github.com/manifold-inc/targon',
    # 'subnet-5': 'https://github.com/OpenKaito/openkaito',
    # 'subnet-6': 'https://github.com/gitphantomman/d3_subnet',
    # 'subnet-7': 'https://github.com/eclipsevortex/SubVortex',
    # 'subnet-8': 'https://github.com/taoshidev/proprietary-trading-network',
    # 'subnet-9': 'https://github.com/macrocosm-os/pretraining',
    # 'subnet-10': 'https://github.com/Sturdy-Subnet/sturdy-subnet',
    # 'subnet-11': 'https://github.com/impel-intelligence/dippy-bittensor-subnet',
    # 'subnet-12': 'https://github.com/backend-developers-ltd/ComputeHorde',
    # 'subnet-13': 'https://github.com/macrocosm-os/data-universe',
    # 'subnet-14': 'https://github.com/synapsec-ai/llm-defender-subnet',
    # 'subnet-15': 'https://github.com/blockchain-insights/blockchain-data-subnet-ops',
    # 'subnet-16': 'https://github.com/eseckft/BitAds.ai',
    # 'subnet-17': 'https://github.com/404-Repo/three-gen-subnet',
    # 'subnet-18': 'https://github.com/corcel-api/cortex.t',
    # 'subnet-20': 'https://github.com/RogueTensor/bitagent_subnet',
    # 'subnet-21': 'https://github.com/ifrit98/storage-subnet',
    # 'subnet-22': 'https://github.com/Datura-ai/smart-scrape',
    # 'subnet-23': 'https://github.com/NicheTensor/NicheImage',
    # 'subnet-24': 'https://github.com/omegalabsinc/omegalabs-bittensor-subnet',
    # 'subnet-25': 'https://github.com/macrocosm-os/folding',
    # 'subnet-26': 'https://github.com/TensorAlchemy/TensorAlchemy',
    # 'subnet-27': 'https://github.com/neuralinternet/compute-subnet',
    # 'subnet-28': 'https://github.com/foundryservices/snpOracle',
    # 'subnet-29': 'https://github.com/fractal-net/fractal',
    # 'subnet-30': 'https://github.com/womboai/wombo-bittensor-subnet',
    # 'subnet-31': 'https://github.com/nimaaghli/NASChain',
    # 'subnet-32': 'https://github.com/It-s-AI/llm-detection',
    # 'subnet-33': 'https://github.com/afterpartyai/bittensor-conversation-genome-project',
    # 'subnet-34': 'https://github.com/Healthi-Labs/healthi-subnet',
    # subnet-35'
    # 'subnet-36': 'https://github.com/HIP-Labs/HIP-Subnet'
}


def validate_crawl_targets(targets):
    """Load and validate the list of crawl targets.

    Args:
        targets (dict): A dictionary containing crawl targets information. The keys are the
        repository IDs and the values are the details of each repo.

    Returns:
        list: A list of validated crawl targets.
    """
    validated_targets = []

    for repo_id, details in targets.items():
        details['repo_id'] = repo_id
        validated_targets.append(RepoCrawlTarget.model_validate(details))

    return validated_targets


crawl_targets = validate_crawl_targets(_crawl_targets)

"""Versioned evaluation targets and shared molecular prompt formatting."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EnergyProtocol:
    descriptor: str
    solvent: str
    forcefield: str
    cutoff_nm: float | None = None
    dielectric: float | None = None


PROTOCOLS = {
    p.descriptor: p for p in (
        EnergyProtocol('ff14sb-obc1-2nm', 'implicit_water', 'ff14sb', 2.0),
        EnergyProtocol('ff14sb', 'vacuum', 'ff14sb'),
        EnergyProtocol('ff14sb-tip3p-rf2nm', 'explicit_water', 'ff14sb', 2.0, 78.5),
        EnergyProtocol('gaff2-am1bcc', 'vacuum', 'gaff2-am1bcc'),
        EnergyProtocol('gaff2-am1bcc-obc1-2nm', 'implicit_water', 'gaff2-am1bcc', 2.0),
        EnergyProtocol('gaff2-am1bcc-tip3p-rf2nm', 'explicit_water', 'gaff2-am1bcc', 2.0, 78.5),
        EnergyProtocol('ff14sb-obc1-1nm', 'implicit_water', 'ff14sb', 1.0),
    )
}


def environment_description(target_id: str, n_waters: int = 0) -> str:
    """Temperature is added by the tokenizer; no duplicated K field here."""
    protocol = PROTOCOLS[target_id]
    if type(n_waters) is not int or n_waters < 0:
        raise ValueError('n_waters must be a nonnegative integer')
    if protocol.solvent == 'explicit_water':
        if n_waters == 0:
            raise ValueError('explicit-water prompts require at least one water')
        environment = f'{n_waters} water'
    else:
        if n_waters:
            raise ValueError('non-explicit protocol cannot contain water molecules')
        environment = 'water' if protocol.solvent == 'implicit_water' else 'vacuum'
    return f'{environment}, {protocol.descriptor}'
